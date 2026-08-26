import argparse
import math
import csv
import os
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaac_so_arm101.tasks
from isaaclab_tasks.utils import parse_env_cfg
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK = "Isaac-SO-ARM101-RGBBlocks-Play-v0"
DATA_DIR = os.path.expanduser("~/robot_arm/isaac_so_arm101/my_scripts/results/traj_data")
FIG_DIR = os.path.expanduser("~/robot_arm/isaac_so_arm101/my_scripts/results/figures")

JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]

FIXED = {
    "shoulder_lift": {"armature": 0.197, "fric_s": 0.001, "fric_v": 0.007},
    "elbow_flex":    {"armature": 0.073, "fric_s": 0.867, "fric_v": 0.023},
    "wrist_flex":    {"armature": 0.111, "fric_s": 0.386, "fric_v": 0.109},
}

INIT = {
    "shoulder_lift": {"stiffness": 404.0, "damping": 59.0},
    "elbow_flex":    {"stiffness": 238.0, "damping": 35.0},
    "wrist_flex":    {"stiffness": 272.0, "damping": 45.0},
}

SCALE = np.array([500.0, 100.0] * 3)
SETTLE_STEPS = 20   # 궤적 시작 전 안정화 스텝

BASE_POSE_DEG = {
    "shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
    "wrist_flex": 0.0, "wrist_roll": 0.0,
}
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
PHASE = {"shoulder_lift": 0.0, "elbow_flex": 2.094, "wrist_flex": 4.189}
RATE = 50
STEP_AMP = 15.0
SIN_AMP = 20.0
SIN_FREQ = 0.2
DURATION = 13 + 3 / SIN_FREQ


def offset(t, joint):
    ph = PHASE[joint]
    if t < 2:
        return 0.0
    elif t < 5:
        return -STEP_AMP
    elif t < 8:
        return 0.0
    elif t < 11:
        return STEP_AMP
    elif t < 13:
        return 0.0
    else:
        return SIN_AMP * math.sin(2 * math.pi * SIN_FREQ * (t - 13) + ph)


def load_real():
    t = []
    pos = {j: [] for j in JOINTS}
    with open(f"{DATA_DIR}/real_multi.csv") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {j: header.index(f"{j}_pos") for j in JOINTS}
        for row in r:
            t.append(float(row[0]))
            for j in JOINTS:
                pos[j].append(float(row[idx[j]]))
    t = np.array(t)
    m = t <= DURATION
    t = t[m]
    for j in JOINTS:
        pos[j] = np.array(pos[j])[m]
    return t, pos


def main():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs,
                        use_fabric=not args.disable_fabric)
    cfg.episode_length_s = 1e9
    cfg.actions.arm_action.scale = 1.0
    cfg.actions.arm_action.use_default_offset = False
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    names = robot.data.joint_names
    j_ix = {j: names.index(j) for j in JOINTS}

    real_t, real_pos = load_real()
    real_rel = {j: real_pos[j] - real_pos[j][0] for j in JOINTS}

    # 초기 자세 텐서 (미리 구성)
    q_init = robot.data.default_joint_pos.clone()
    for jn, d in BASE_POSE_DEG.items():
        q_init[0, names.index(jn)] = math.radians(d)
    qd_zero = torch.zeros_like(q_init)
    base_action = torch.tensor(
        [[math.radians(BASE_POSE_DEG[jj]) for jj in ARM] + [-1.0]],
        dtype=torch.float32, device=env.device)

    def apply_fixed():
        for j in JOINTS:
            f = FIXED[j]
            jid = [j_ix[j]]
            robot.write_joint_armature_to_sim(f["armature"], joint_ids=jid)
            robot.write_joint_friction_coefficient_to_sim(
                f["fric_s"], f["fric_s"], f["fric_v"], joint_ids=jid)

    def apply_sd(theta):
        for i, j in enumerate(JOINTS):
            jid = [j_ix[j]]
            robot.write_joint_stiffness_to_sim(float(theta[2 * i]), joint_ids=jid)
            robot.write_joint_damping_to_sim(float(theta[2 * i + 1]), joint_ids=jid)

    def simulate(theta):
        env.reset()
        apply_fixed()
        apply_sd(theta)
        # 명시적 초기화: 위치+속도0
        robot.write_joint_state_to_sim(q_init.clone(), qd_zero.clone())
        # 안정화: 목표 자세로 여러 스텝 (이전 상태 잔류 제거)
        for _ in range(SETTLE_STEPS):
            env.step(base_action)
            robot.write_joint_state_to_sim(q_init.clone(), qd_zero.clone())

        sim_t = []
        sim_pos = {j: [] for j in JOINTS}
        t = 0.0
        dt = 1.0 / RATE
        while t <= DURATION:
            arm_deg = dict(BASE_POSE_DEG)
            for j in JOINTS:
                arm_deg[j] = BASE_POSE_DEG[j] + offset(t, j)
            arm_rad = [math.radians(arm_deg[jj]) for jj in ARM]
            action = torch.tensor([arm_rad + [-1.0]], dtype=torch.float32, device=env.device)
            env.step(action)
            sim_t.append(t)
            for j in JOINTS:
                sim_pos[j].append(math.degrees(robot.data.joint_pos[0, j_ix[j]].item()))
            t += dt
        return np.array(sim_t), {j: np.array(sim_pos[j]) for j in JOINTS}

    def rmse_all(theta):
        st, sp = simulate(theta)
        total = 0.0
        per = {}
        for j in JOINTS:
            si = np.interp(real_t, st, sp[j])
            si_rel = si - si[0]
            e = float(np.sqrt(np.mean((real_rel[j] - si_rel) ** 2)))
            per[j] = e
            total += e ** 2
        return math.sqrt(total / len(JOINTS)), per

    def cost(x):
        xc = np.clip(x, 0.0, 1.0)
        theta = xc * SCALE
        r, per = rmse_all(theta)
        print(f"  k/c={np.round(theta,1)}  RMSE={r:.4f}")
        return r

    theta0 = np.array([
        INIT["shoulder_lift"]["stiffness"], INIT["shoulder_lift"]["damping"],
        INIT["elbow_flex"]["stiffness"], INIT["elbow_flex"]["damping"],
        INIT["wrist_flex"]["stiffness"], INIT["wrist_flex"]["damping"],
    ])
    x0 = theta0 / SCALE

    print("\n[multi] stiffness/damping 6개 최적화 (안정화 스텝 추가)")
    print("=== 재현성 확인: 초기값 3회 반복 ===")
    for rep in range(3):
        r, per = rmse_all(theta0)
        print(f"  rep{rep}: RMSE={r:.4f}  per={ {k: round(v,2) for k,v in per.items()} }")

    base_rmse, base_per = rmse_all(theta0)

    print("\n=== 최적화 (Powell) ===")
    res = minimize(cost, x0, method="Powell", bounds=[(0, 1)] * 6,
                   options={"maxiter": 30, "xtol": 1e-3, "ftol": 1e-3})
    theta_opt = np.clip(res.x, 0, 1) * SCALE
    tuned_rmse, tuned_per = rmse_all(theta_opt)

    print(f"\n=== 결과 ===")
    print(f"단일튜닝 초기 RMSE = {base_rmse:.4f} deg  per={ {k: round(v,2) for k,v in base_per.items()} }")
    print(f"다중튜닝 후   RMSE = {tuned_rmse:.4f} deg  per={ {k: round(v,2) for k,v in tuned_per.items()} }")
    print(f"감소율 {(1-tuned_rmse/base_rmse)*100:.1f}%")
    for i, j in enumerate(JOINTS):
        print(f"  {j:14s}: stiffness {theta0[2*i]:.0f}->{theta_opt[2*i]:.1f}, "
              f"damping {theta0[2*i+1]:.0f}->{theta_opt[2*i+1]:.1f}")

    st_b, sp_b = simulate(theta0)
    st_t, sp_t = simulate(theta_opt)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, j in zip(axes, JOINTS):
        sb = np.interp(real_t, st_b, sp_b[j]); sb -= sb[0]
        stt = np.interp(real_t, st_t, sp_t[j]); stt -= stt[0]
        ax.plot(real_t, real_rel[j], color="tab:red", label="real")
        ax.plot(real_t, sb, color="tab:blue", alpha=0.6, label="sim single-tuned")
        ax.plot(real_t, stt, color="tab:green", label="sim multi-tuned")
        ax.set_title(f"{j}  (single {base_per[j]:.2f} -> multi {tuned_per[j]:.2f})")
        ax.set_ylabel("angle rel (deg)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Multi-joint coupled trajectory: single vs multi tuning")
    fig.tight_layout()
    out = f"{FIG_DIR}/optimize_multi.png"
    fig.savefig(out, dpi=80)
    print(f"\n저장: {out}")

    env.close()
    simulation_app.close()


main()

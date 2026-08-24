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

JOINT = "shoulder_lift"

BASE_PARAMS = {
    "shoulder_lift": {"stiffness": 170.0, "damping": 65.0},
    "elbow_flex": {"stiffness": 120.0, "damping": 45.0},
    "wrist_flex": {"stiffness": 80.0, "damping": 30.0},
}

# 파라미터 스케일 (정규화용 상한): stiffness, damping, armature, fric_s, fric_v
SCALE = np.array([400.0, 200.0, 0.5, 1.0, 0.5])

BASE_POSE_DEG = {
    "shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
    "wrist_flex": 0.0, "wrist_roll": 0.0,
}
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
RATE = 50
STEP_AMP = 15.0
SIN_AMP = 20.0
SIN_FREQ = 0.2
SIN_CYCLES = 2
DURATION = 13 + SIN_CYCLES / SIN_FREQ


def target_offset(t):
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
        return SIN_AMP * math.sin(2 * math.pi * SIN_FREQ * (t - 13))


def load_real(joint):
    t, pos = [], []
    with open(f"{DATA_DIR}/real_{joint}.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t.append(float(row[0]))
            pos.append(float(row[2]))
    t = np.array(t)
    pos = np.array(pos)
    m = t <= DURATION
    return t[m], pos[m]


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
    j_ix = names.index(JOINT)

    real_t, real_pos = load_real(JOINT)
    real_rel = real_pos - real_pos[0]

    def apply_params(theta):
        stiffness, damping, armature, fric_s, fric_v = theta
        jid = [j_ix]
        robot.write_joint_stiffness_to_sim(float(stiffness), joint_ids=jid)
        robot.write_joint_damping_to_sim(float(damping), joint_ids=jid)
        robot.write_joint_armature_to_sim(float(armature), joint_ids=jid)
        robot.write_joint_friction_coefficient_to_sim(
            float(fric_s), float(fric_s), float(fric_v), joint_ids=jid)

    def simulate(theta):
        env.reset()
        apply_params(theta)
        q = robot.data.default_joint_pos.clone()
        for jn, d in BASE_POSE_DEG.items():
            q[0, names.index(jn)] = math.radians(d)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))

        sim_t, sim_pos = [], []
        t = 0.0
        dt = 1.0 / RATE
        while t <= DURATION:
            off = target_offset(t)
            arm_deg = dict(BASE_POSE_DEG)
            arm_deg[JOINT] = BASE_POSE_DEG[JOINT] + off
            arm_rad = [math.radians(arm_deg[j]) for j in ARM]
            action = torch.tensor([arm_rad + [-1.0]], dtype=torch.float32, device=env.device)
            env.step(action)
            sim_t.append(t)
            sim_pos.append(math.degrees(robot.data.joint_pos[0, j_ix].item()))
            t += dt
        return np.array(sim_t), np.array(sim_pos)

    def rmse_of(theta):
        st, sp = simulate(theta)
        sp_on_real = np.interp(real_t, st, sp)
        sp_rel = sp_on_real - sp_on_real[0]
        return float(np.sqrt(np.mean((real_rel - sp_rel) ** 2)))

    # 정규화된 x(0~1)를 받아 실제 theta로 복원, bounds는 clip
    def cost(x):
        xc = np.clip(x, 0.0, 1.0)
        theta = xc * SCALE
        r = rmse_of(theta)
        print(f"  theta={np.round(theta,4)}  RMSE={r:.4f}")
        return r

    bp = BASE_PARAMS[JOINT]
    theta0 = np.array([bp["stiffness"], bp["damping"], 0.01, 0.01, 0.01])
    x0 = theta0 / SCALE

    print(f"\n[{JOINT}] 최적화 시작 (headless, DURATION={DURATION:.0f}s)")
    print("=== base RMSE ===")
    base_rmse = rmse_of(theta0)
    print(f"  base RMSE = {base_rmse:.4f}")

    print("\n=== 최적화 (Powell, bounds 0~1) ===")
    res = minimize(cost, x0, method="Powell",
                   bounds=[(0, 1)] * 5,
                   options={"maxiter": 40, "xtol": 1e-3, "ftol": 1e-3})
    theta_opt = np.clip(res.x, 0, 1) * SCALE
    tuned_rmse = res.fun

    print(f"\n=== 결과 ({JOINT}) ===")
    print(f"base  RMSE = {base_rmse:.4f} deg")
    print(f"tuned RMSE = {tuned_rmse:.4f} deg  ({(1-tuned_rmse/base_rmse)*100:.1f}% 감소)")
    labels = ["stiffness", "damping", "armature", "friction_s", "friction_v"]
    for l, v in zip(labels, theta_opt):
        print(f"  {l:12s}: {v:.4f}")

    st_b, sp_b = simulate(theta0)
    st_t, sp_t = simulate(theta_opt)
    sp_b_r = np.interp(real_t, st_b, sp_b); sp_b_r -= sp_b_r[0]
    sp_t_r = np.interp(real_t, st_t, sp_t); sp_t_r -= sp_t_r[0]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(real_t, real_rel, "-", color="tab:red", label="real")
    ax.plot(real_t, sp_b_r, "-", color="tab:blue", alpha=0.6,
            label=f"sim base (RMSE={base_rmse:.2f})")
    ax.plot(real_t, sp_t_r, "-", color="tab:green",
            label=f"sim tuned (RMSE={tuned_rmse:.2f})")
    ax.set_title(f"{JOINT}: trajectory match")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angle rel. (deg)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = f"{FIG_DIR}/optimize_{JOINT}.png"
    fig.savefig(out, dpi=80)
    print(f"\n저장: {out}")

    env.close()
    simulation_app.close()


main()

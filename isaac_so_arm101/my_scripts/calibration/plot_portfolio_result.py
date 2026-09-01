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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK = "Isaac-SO-ARM101-RGBBlocks-Play-v0"
DATA_DIR = os.path.expanduser("~/robot_arm/isaac_so_arm101/my_scripts/results/traj_data")
FIG_DIR = os.path.expanduser("~/robot_arm/isaac_so_arm101/my_scripts/results/figures")

JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]
JOINT_LABEL = {"shoulder_lift": "Shoulder", "elbow_flex": "Elbow", "wrist_flex": "Wrist"}

BASE = {
    "shoulder_lift": [170.0, 65.0, 0.0, 0.0, 0.0],
    "elbow_flex":    [120.0, 45.0, 0.0, 0.0, 0.0],
    "wrist_flex":    [80.0, 30.0, 0.0, 0.0, 0.0],
}
TUNED = {
    "shoulder_lift": [418.0, 58.0, 0.197, 0.001, 0.007],
    "elbow_flex":    [269.0, 37.0, 0.073, 0.867, 0.023],
    "wrist_flex":    [390.0, 45.0, 0.111, 0.386, 0.109],
}

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
SETTLE_STEPS = 20
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

    q_init = robot.data.default_joint_pos.clone()
    for jn, d in BASE_POSE_DEG.items():
        q_init[0, names.index(jn)] = math.radians(d)
    qd_zero = torch.zeros_like(q_init)
    base_action = torch.tensor(
        [[math.radians(BASE_POSE_DEG[jj]) for jj in ARM] + [-1.0]],
        dtype=torch.float32, device=env.device)

    def apply_params(PARAMS):
        for j in JOINTS:
            k, c, a, fs, fv = PARAMS[j]
            jid = [j_ix[j]]
            robot.write_joint_stiffness_to_sim(k, joint_ids=jid)
            robot.write_joint_damping_to_sim(c, joint_ids=jid)
            robot.write_joint_armature_to_sim(a, joint_ids=jid)
            robot.write_joint_friction_coefficient_to_sim(fs, fs, fv, joint_ids=jid)

    def simulate(PARAMS):
        env.reset()
        apply_params(PARAMS)
        robot.write_joint_state_to_sim(q_init.clone(), qd_zero.clone())
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

    def rmse(real_r, sim_t, sim_p):
        si = np.interp(real_t, sim_t, sim_p)
        si -= si[0]
        return float(np.sqrt(np.mean((real_r - si) ** 2))), si

    st_b, sp_b = simulate(BASE)
    st_t, sp_t = simulate(TUNED)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.edgecolor": "#333333", "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.7,
        "axes.axisbelow": True,
    })
    C_REAL = "#C0392B"; C_BASE = "#AAB7C4"; C_TUNED = "#2E8B57"

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for ax, j in zip(axes, JOINTS):
        rb, sb = rmse(real_rel[j], st_b, sp_b[j])
        rt, stt = rmse(real_rel[j], st_t, sp_t[j])

        ax.axvspan(0, 13, color="#F7F7F7", zorder=0)
        ax.axvspan(13, DURATION, color="#F0F5FA", zorder=0)

        ax.plot(real_t, real_rel[j], color=C_REAL, lw=2.4, label="Real", zorder=5)
        ax.plot(real_t, sb, color=C_BASE, lw=1.8, ls="--",
                label=f"Sim base  (RMSE {rb:.2f}\u00b0)", zorder=3)
        ax.plot(real_t, stt, color=C_TUNED, lw=2.0,
                label=f"Sim tuned (RMSE {rt:.2f}\u00b0)", zorder=4)

        ax.set_ylabel(f"{JOINT_LABEL[j]}\n angle (deg)", fontsize=11)
        ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9, ncol=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, DURATION)

    axes[0].text(6.5, axes[0].get_ylim()[1]*0.86, "Step response",
                 ha="center", fontsize=10, color="#999999", style="italic")
    axes[0].text(20, axes[0].get_ylim()[1]*0.86, "Sinusoid tracking",
                 ha="center", fontsize=10, color="#7A96B0", style="italic")

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.suptitle("Robot Arm Dynamics Matching: Real vs Sim Trajectory",
                 fontsize=14, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{FIG_DIR}/portfolio_traj.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\n저장: {out}")

    print("\n=== RMSE (복합 동작) ===")
    for j in JOINTS:
        rb, _ = rmse(real_rel[j], st_b, sp_b[j])
        rt, _ = rmse(real_rel[j], st_t, sp_t[j])
        print(f"  {JOINT_LABEL[j]:10s}: base {rb:.2f} -> tuned {rt:.2f}")

    env.close()
    simulation_app.close()


main()

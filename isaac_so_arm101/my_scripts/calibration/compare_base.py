import argparse
import math
import csv
import os
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
import numpy as np
import gymnasium as gym
import isaac_so_arm101.tasks
from isaaclab_tasks.utils import parse_env_cfg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK = "Isaac-SO-ARM101-RGBBlocks-Play-v0"
DATA_DIR = os.path.expanduser("~/robot_arm/isaac_so_arm101/my_scripts/results/traj_data")
JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]

BASE_POSE_DEG = {
    "shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
    "wrist_flex": 0.0, "wrist_roll": 0.0,
}
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
RATE = 50
STEP_AMP = 15.0
SIN_AMP = 20.0
SIN_FREQ = 0.2
DURATION = 13 + 3 / SIN_FREQ


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
    t, target, pos = [], [], []
    with open(f"{DATA_DIR}/real_{joint}.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t.append(float(row[0]))
            target.append(float(row[1]))
            pos.append(float(row[2]))
    return np.array(t), np.array(target), np.array(pos)


def simulate(env, robot, names, joint):
    arm_ix = [names.index(j) for j in ARM]
    grip_ix = names.index("gripper")
    j_ix = names.index(joint)

    q = robot.data.default_joint_pos.clone()
    for jn, d in BASE_POSE_DEG.items():
        q[0, names.index(jn)] = math.radians(d)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))

    sim_t, sim_pos = [], []
    t = 0.0
    dt = 1.0 / RATE
    while t <= DURATION:
        off = target_offset(t)
        arm_target_deg = dict(BASE_POSE_DEG)
        arm_target_deg[joint] = BASE_POSE_DEG[joint] + off
        arm_rad = [math.radians(arm_target_deg[j]) for j in ARM]
        action = torch.tensor([arm_rad + [-1.0]], dtype=torch.float32, device=env.device)
        env.step(action)
        sim_t.append(t)
        sim_pos.append(math.degrees(robot.data.joint_pos[0, j_ix].item()))
        t += dt
    return np.array(sim_t), np.array(sim_pos)


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

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    print(f"\n{'joint':16s} {'RMSE(deg)':>10}")
    for ax, joint in zip(axes, JOINTS):
        env.reset()
        rt, rtar, rpos = load_real(joint)
        st, spos = simulate(env, robot, names, joint)

        sim_on_real = np.interp(rt, st, spos)
        rmse = float(np.sqrt(np.mean((rpos - sim_on_real) ** 2)))
        print(f"{joint:16s} {rmse:>10.3f}")

        ax.plot(rt, rtar, "--", color="gray", label="target")
        ax.plot(rt, rpos, "-", color="tab:red", label="real")
        ax.plot(st, spos, "-", color="tab:blue", label="sim (base)")
        ax.set_title(f"{joint}   RMSE={rmse:.2f} deg")
        ax.set_ylabel("angle (deg)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    out = os.path.expanduser("~/robot_arm/isaac_so_arm101/my_scripts/results/figures/compare_base.png")
    fig.savefig(out, dpi=80)
    print(f"\n저장: {out}")

    env.close()
    simulation_app.close()


main()

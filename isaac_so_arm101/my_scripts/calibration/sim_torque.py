import argparse
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

TASK = "Isaac-SO-ARM101-RGBBlocks-Play-v0"
JOINT = "shoulder_lift"

TARGET_POSE_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 85.0,
    "elbow_flex": -85.0,
    "wrist_flex": 15.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}

WEIGHTS_G = [0, 20, 50, 100, 150, 170]
GRIPPER_LINK = "gripper_link"
SETTLE_STEPS = 60


def main():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs,
                        use_fabric=not args.disable_fabric)
    cfg.episode_length_s = 1e9
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    names = robot.data.joint_names
    joint_ix = names.index(JOINT)

    body_names = robot.data.body_names
    grip_body_ix = body_names.index(GRIPPER_LINK)

    import math
    q = robot.data.default_joint_pos.clone()
    for j, deg in TARGET_POSE_DEG.items():
        q[0, names.index(j)] = math.radians(deg)

    base_masses = robot.root_physx_view.get_masses().clone()
    orig_grip_mass = base_masses[0, grip_body_ix].item()

    print(f"[{JOINT}] 시뮬 토크 측정")
    print(f"기준 gripper 질량: {orig_grip_mass:.4f} kg")
    print(f"{'무게(g)':>8} {'applied_torque(N·m)':>20}")

    action = torch.zeros((1, env.action_space.shape[1]), device=env.device)

    for grams in WEIGHTS_G:
        masses = base_masses.clone()
        masses[0, grip_body_ix] = orig_grip_mass + grams / 1000.0
        robot.root_physx_view.set_masses(masses, torch.arange(1))

        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        for _ in range(SETTLE_STEPS):
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            env.step(action)

        tau = robot.data.applied_torque[0, joint_ix].item()
        print(f"{grams:>8} {tau:>20.4f}")

    print("\n완료. Ctrl+C로 종료.")
    try:
        while simulation_app.is_running():
            env.step(action)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        simulation_app.close()


main()

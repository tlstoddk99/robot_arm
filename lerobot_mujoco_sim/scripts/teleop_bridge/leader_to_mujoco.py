"""
Connect the physical SO-101 leader arm to the SO-101 MuJoCo sim and mirror
its joint positions in real time.

Usage:
    python scripts/teleop_bridge/leader_to_mujoco.py \
        --leader-port /dev/ttyACM0 \
        --leader-id my_leader_arm \
        --xml-path asset/scene_so101_y.xml
"""

import argparse
import time

import numpy as np

from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig
from mujoco_env.y_env import SimpleEnv

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_INVERTED = False  # flip to True if open/close looks backwards in sim


def leader_action_to_sim_action(action: dict, env: SimpleEnv) -> np.ndarray:
    arm_rad = np.array([np.radians(action[f"{j}.pos"]) for j in ARM_JOINTS], dtype=np.float32)
    arm_rad = np.clip(arm_rad, env.joint_mins, env.joint_maxs)

    gripper_raw = action["gripper.pos"] / 100.0
    gripper_frac = gripper_raw if not GRIPPER_INVERTED else (1.0 - gripper_raw)
    gripper_frac = float(np.clip(gripper_frac, 0.0, 1.0))

    return np.concatenate([arm_rad, [gripper_frac]]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--leader-id", required=True)
    parser.add_argument("--xml-path", default="asset/scene_so101_y.xml")
    parser.add_argument("--hz", type=float, default=50.0)
    args = parser.parse_args()

    leader_cfg = SOLeaderTeleopConfig(port=args.leader_port, id=args.leader_id)
    leader = SO101Leader(leader_cfg)
    leader.connect()
    print(f"[leader] connected on {args.leader_port} (id={args.leader_id})")

    env = SimpleEnv(
        robot_profile="so101",
        xml_path=args.xml_path,
        action_type="joint_angle",
        state_type="joint_angle",
    )
    env.reset()
    env.init_viewer()
    print("[sim] MuJoCo viewer ready. Move the leader arm to mirror it in sim.")

    period = 1.0 / args.hz
    try:
        while True:
            t0 = time.perf_counter()

            action = leader.get_action()
            sim_action = leader_action_to_sim_action(action, env)
            env.step(sim_action)
            env.render()

            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[teleop_bridge] stopping.")
    finally:
        leader.disconnect()


if __name__ == "__main__":
    main()
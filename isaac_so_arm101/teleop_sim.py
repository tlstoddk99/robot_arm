"""Leader 팔(흰색, ttyACM1) → 시뮬 SO-101 텔레오프 (1단계: 시뮬만)
   방법1: arm action을 순수 위치제어(scale=1, offset=off)로 override"""
import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-SO-ARM101-RGBBlocks-Play-v0")
parser.add_argument("--leader_port", default="/dev/ttyACM1")
parser.add_argument("--leader_id", default="my_leader")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", help="관절값 출력")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaac_so_arm101.tasks
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

# arm 5개 (Isaac arm_action 순서) + gripper는 별도(Binary)
ARM_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# 관절별 부호 (관찰: shoulder_lift, elbow_flex 반대)
JOINT_SIGN = {
    "shoulder_pan":  1.0,
    "shoulder_lift": 1.0,
    "elbow_flex":    1.0,
    "wrist_flex":    1.0,
    "wrist_roll":    1.0,
}
# 영점 offset (deg) — 필요시 조정
JOINT_OFFSET = {
    "shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
    "wrist_flex":   0.0, "wrist_roll":    0.0,
}
GRIP_THRESH = 45.0   # leader gripper 이 각도 넘으면 열림

_prev_deg = {}
def _unwrap(name, deg):
    if name in _prev_deg:
        d = deg - _prev_deg[name]
        if d > 180: deg -= 360
        elif d < -180: deg += 360
    _prev_deg[name] = deg
    return deg

def leader_to_action(leader_dict, device, debug=False):
    vals = []
    for j in ARM_ORDER:
        deg = leader_dict.get(f"{j}.pos", 0.0)
        deg = _unwrap(j, deg)
        rad = math.radians(deg - JOINT_OFFSET[j]) * JOINT_SIGN[j]
        vals.append(rad)
    # gripper: Binary action (양수=열림, 음수=닫힘)
    grip_deg = leader_dict.get("gripper.pos", 0.0)
    grip_cmd = 1.0 if grip_deg > GRIP_THRESH else -1.0
    vals.append(grip_cmd)
    if debug:
        print(" | ".join(f"{j}:{v:6.2f}" for j, v in zip(ARM_ORDER + ["grip"], vals)))
    return torch.tensor([vals], dtype=torch.float32, device=device)

def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device,
        num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.episode_length_s = 100000.0

    # 방법1: arm action을 순수 절대위치 제어로 override (RL용 scale/offset 제거)
    env_cfg.actions.arm_action.scale = 1.0
    env_cfg.actions.arm_action.use_default_offset = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    cfg = SO101LeaderConfig(port=args_cli.leader_port, id=args_cli.leader_id)
    leader = SO101Leader(cfg)
    leader.connect()
    print("[teleop] leader 연결됨. 팔을 움직이면 시뮬이 따라갑니다. (Ctrl+C 종료)")

    env.reset()
    try:
        while simulation_app.is_running():
            leader_dict = leader.get_action()
            action = leader_to_action(leader_dict, env.device, args_cli.debug)
            env.step(action)
    except KeyboardInterrupt:
        print("\n[teleop] 종료")
    finally:
        leader.disconnect()
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()

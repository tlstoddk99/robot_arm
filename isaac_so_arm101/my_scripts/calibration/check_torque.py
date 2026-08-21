"""시뮬 로봇의 관절 상태·토크 읽기 확인 (정적 토크 실험 준비)"""
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

def main():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs,
                        use_fabric=not args.disable_fabric)
    cfg.episode_length_s = 1e9
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    names = robot.data.joint_names
    print("\n===== 관절 순서 =====")
    print(names)

    # 몇 스텝 진행 후 상태 읽기 (안정화)
    action = torch.zeros((1, env.action_space.shape[1]), device=env.device)
    for _ in range(30):
        env.step(action)

    print("\n===== 읽을 수 있는 데이터 =====")
    d = robot.data
    for attr in ["joint_pos", "joint_vel", "applied_torque", "computed_torque"]:
        try:
            val = getattr(d, attr)
            print(f"[OK] {attr}: shape={tuple(val.shape)}")
            print(f"      {val[0].cpu().numpy().round(4)}")
        except Exception as e:
            print(f"[X]  {attr}: {e}")

    # 링크 질량 확인
    print("\n===== 링크 질량 (시뮬 내부) =====")
    try:
        masses = robot.root_physx_view.get_masses()
        print(f"link masses: {masses[0].cpu().numpy().round(4)}")
        print(f"총 질량: {masses[0].sum().item():.4f} kg")
    except Exception as e:
        print(f"질량 읽기 실패: {e}")

    print("\n확인 완료. Ctrl+C로 종료.")
    try:
        while simulation_app.is_running():
            env.step(action)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()

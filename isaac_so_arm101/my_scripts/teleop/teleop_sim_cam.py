"""Leader(흰색,ACM1) → 실물 follower(검정,ACM0) + 시뮬 SO-101 동시 텔레오프
   + 시뮬/실제 카메라 4개 뷰 실시간 창 표시 (저장은 옵션)"""
import argparse
import math
import os
from isaaclab.app import AppLauncher

# ===== 설정 =====
TASK        = "Isaac-SO-ARM101-RGBBlocks-Play-v0"
LEADER_PORT   = "/dev/ttyACM1"
LEADER_ID     = "my_leader"
FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID   = "my_follower"

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SIGN   = {"shoulder_pan": 1, "shoulder_lift": 1, "elbow_flex": 1, "wrist_flex": 1, "wrist_roll": 1}
OFFSET = {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 15, "wrist_roll": -90}
GRIP_OPEN_DEG = 45.0
GRIP_OPEN, GRIP_CLOSE = 0.5, 0.0

SIM_CAMS  = {"top_cam": "top", "gripper_cam": "gripper"}   # 시뮬 카메라
REAL_CAMS = {"top": 2, "gripper": 4}                        # video2=탑뷰, video4=그리퍼
CAM_W, CAM_H, CAM_FPS = 640, 480, 30
SAVE_DIR   = "frames"
VIEW_EVERY = 3     # N 스텝마다 창 갱신 (너무 자주면 느림)
SAVE_EVERY = 10
# ================

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--no_follower", action="store_true")
parser.add_argument("--save_cam", action="store_true", help="이미지도 저장")
parser.add_argument("--debug", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
import isaac_so_arm101.tasks
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import signal

try:
    import imageio.v2 as imageio
except Exception:
    import imageio

_prev = {}
def _unwrap(name, deg):
    if name in _prev:
        d = deg - _prev[name]
        if d > 180: deg -= 360
        elif d < -180: deg += 360
    _prev[name] = deg
    return deg

def arm_rad(ld):
    return [math.radians(_unwrap(j, ld.get(f"{j}.pos", 0.0)) - OFFSET[j]) * SIGN[j] for j in ARM]

def grip_open(ld):
    return ld.get("gripper.pos", 0.0) > GRIP_OPEN_DEG

def open_real_cams():
    caps = {}
    for name, idx in REAL_CAMS.items():
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        caps[name] = cap
        print(f"[cam] real '{name}' (video{idx}) opened={cap.isOpened()}")
    return caps

def get_sim_img(env, cam):
    rgb = env.scene[cam].data.output["rgb"]
    return rgb[0, ..., :3].detach().cpu().numpy().astype("uint8")

def get_real_img(cap):
    ok, frame = cap.read()
    if not ok:
        return np.zeros((CAM_H, CAM_W, 3), np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

class Viewer:
    """2x2 실시간 창: [sim_top, real_top] / [sim_gripper, real_gripper]"""
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(2, 2, figsize=(10, 7))
        self.titles = [["SIM top", "REAL top"], ["SIM gripper", "REAL gripper"]]
        self.im = [[None, None], [None, None]]
        blank = np.zeros((CAM_H, CAM_W, 3), np.uint8)
        for r in range(2):
            for c in range(2):
                self.im[r][c] = self.ax[r][c].imshow(blank)
                self.ax[r][c].set_title(self.titles[r][c])
                self.ax[r][c].axis("off")
        self.fig.tight_layout()

    def update(self, imgs):
        # imgs: dict {"sim_top","real_top","sim_gripper","real_gripper"}
        self.im[0][0].set_data(imgs["sim_top"])
        self.im[0][1].set_data(imgs["real_top"])
        self.im[1][0].set_data(imgs["sim_gripper"])
        self.im[1][1].set_data(imgs["real_gripper"])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

def main():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs,
                        use_fabric=not args.disable_fabric)
    cfg.episode_length_s = 1e9
    cfg.actions.arm_action.scale = 1.0
    cfg.actions.arm_action.use_default_offset = False
    cfg.scene.ee_frame.debug_vis = False
    cfg.commands.object_pose.debug_vis = False
    env = gym.make(TASK, cfg=cfg).unwrapped

    robot = env.scene["robot"]
    names = robot.data.joint_names
    arm_ix = [names.index(j) for j in ARM]
    grip_ix = names.index("gripper")

    real_caps = open_real_cams()
    viewer = Viewer()
    if args.save_cam:
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"[cam] 저장 켜짐: ./{SAVE_DIR}/")

    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID))
    leader.connect()

    follower = None
    if not args.no_follower:
        follower = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID))
        follower.connect()

    def _emergency_stop(signum, frame):
        print("[teleop] 비상 정지 — follower 토크 끄기")
        if follower is not None:
            try:
                follower.bus.disable_torque()
                print("[teleop] 토크 꺼짐")
            except Exception as e:
                print(f"[warn] disable_torque: {e}")
        try: leader.disconnect()
        except: pass
        try:
            if follower is not None: follower.disconnect()
        except: pass
        os._exit(0)
    signal.signal(signal.SIGINT, _emergency_stop)

    env.reset()

    ld = leader.get_action()
    q = robot.data.default_joint_pos.clone()
    for ix, val in zip(arm_ix, arm_rad(ld)):
        q[0, ix] = val
    q[0, grip_ix] = GRIP_OPEN if grip_open(ld) else GRIP_CLOSE
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    print("[teleop] 미러링 + 4뷰 표시 시작 (Ctrl+C 종료)")

    step = 0
    try:
        while simulation_app.is_running():
            ld = leader.get_action()
            if follower is not None:
                follower.send_action(ld)
            vals = arm_rad(ld) + [1.0 if grip_open(ld) else -1.0]
            env.step(torch.tensor([vals], dtype=torch.float32, device=env.device))

            if step % VIEW_EVERY == 0:
                imgs = {
                    "sim_top":     get_sim_img(env, "top_cam"),
                    "sim_gripper": get_sim_img(env, "gripper_cam"),
                    "real_top":     get_real_img(real_caps["top"]),
                    "real_gripper": get_real_img(real_caps["gripper"]),
                }
                viewer.update(imgs)
                if args.save_cam and step % SAVE_EVERY == 0:
                    for k, im in imgs.items():
                        imageio.imwrite(os.path.join(SAVE_DIR, f"{k}_{step:06d}.png"), im)
            step += 1
    except KeyboardInterrupt:
        print("\n[teleop] 종료")
    finally:
        if follower is not None:
            try:
                follower.bus.disable_torque()
                print("[teleop] follower 토크 꺼짐")
            except Exception as e:
                print(f"[warn] disable_torque: {e}")
        for cap in real_caps.values():
            cap.release()
        try: leader.disconnect()
        except: pass
        if follower is not None:
            try: follower.disconnect()
            except: pass
        try: env.close()
        except: pass
        simulation_app.close()

if __name__ == "__main__":
    main()
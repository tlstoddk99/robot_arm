"""Leader(ACM1) -> follower(ACM0) + 시뮬 텔레오프
   MODE(base/tuned) 파라미터 적용, 3관절 real vs sim 실시간 비교
   lerobot식 주기 제어(precise_sleep)로 채터링 방지"""
import argparse
import math
import os
import signal
import time
from collections import deque
from isaaclab.app import AppLauncher

TASK = "Isaac-SO-ARM101-RGBBlocks-Play-v0"
LEADER_PORT = "/dev/ttyACM1"
LEADER_ID = "my_leader"
FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "my_follower"

MODE = "base"   # "base" 또는 "tuned"
FPS = 50         # 목표 제어 주파수 (실물/시뮬 50Hz)

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SIGN = {"shoulder_pan": 1, "shoulder_lift": 1, "elbow_flex": 1, "wrist_flex": 1, "wrist_roll": 1}
OFFSET = {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": -90}
GRIP_OPEN_DEG = 45.0
GRIP_OPEN, GRIP_CLOSE = 0.5, 0.0

PLOT_JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]

PARAMS = {
    "base": {
        "shoulder_lift": [170.0, 65.0, 0.0, 0.0, 0.0],
        "elbow_flex":    [120.0, 45.0, 0.0, 0.0, 0.0],
        "wrist_flex":    [80.0, 30.0, 0.0, 0.0, 0.0],
    },
    "tuned": {
        "shoulder_lift": [404.0, 59.0, 0.197, 0.001, 0.007],
        "elbow_flex":    [238.0, 35.0, 0.073, 0.867, 0.023],
        "wrist_flex":    [272.0, 45.0, 0.111, 0.386, 0.109],
    },
}

WINDOW_SEC = 20.0
READ_EVERY = 2    # N스텝마다 실물 read + 기록
VIEW_EVERY = 10   # N스텝마다 그래프 갱신

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--no_follower", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaac_so_arm101.tasks
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

_prev = {}


def precise_sleep(sec):
    if sec <= 0:
        return
    end = time.perf_counter() + sec
    while time.perf_counter() < end:
        pass


def _unwrap(name, deg):
    if name in _prev:
        d = deg - _prev[name]
        if d > 180:
            deg -= 360
        elif d < -180:
            deg += 360
    _prev[name] = deg
    return deg


def arm_rad(ld):
    return [math.radians(_unwrap(j, ld.get(f"{j}.pos", 0.0)) - OFFSET[j]) * SIGN[j] for j in ARM]


def grip_open(ld):
    return ld.get("gripper.pos", 0.0) > GRIP_OPEN_DEG


class TrajPlot:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        self.fig.suptitle(f"MODE = {MODE}   (real vs sim)")
        self.t = {j: deque() for j in PLOT_JOINTS}
        self.real = {j: deque() for j in PLOT_JOINTS}
        self.sim = {j: deque() for j in PLOT_JOINTS}
        self.lines = {}
        for i, j in enumerate(PLOT_JOINTS):
            ax = self.ax[i]
            self.lines[j] = {
                "real": ax.plot([], [], label="real", color="tab:red")[0],
                "sim": ax.plot([], [], label=f"sim ({MODE})", color="tab:green")[0],
            }
            ax.set_title(j)
            ax.set_ylabel("angle (deg)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
        self.ax[-1].set_xlabel("time (s)")
        self.fig.tight_layout()

    def push(self, joint, t, real_deg, sim_deg):
        self.t[joint].append(t)
        self.real[joint].append(real_deg)
        self.sim[joint].append(sim_deg)
        while self.t[joint] and t - self.t[joint][0] > WINDOW_SEC:
            self.t[joint].popleft()
            self.real[joint].popleft()
            self.sim[joint].popleft()

    def update(self):
        for i, j in enumerate(PLOT_JOINTS):
            ts = list(self.t[j])
            self.lines[j]["real"].set_data(ts, list(self.real[j]))
            self.lines[j]["sim"].set_data(ts, list(self.sim[j]))
            if ts:
                self.ax[i].set_xlim(max(0, ts[-1] - WINDOW_SEC), max(WINDOW_SEC, ts[-1]))
            self.ax[i].relim()
            self.ax[i].autoscale_view(scalex=False)
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
    plot_ix = {j: names.index(j) for j in PLOT_JOINTS}

    env.reset()

    for j in PLOT_JOINTS:
        k, c, a, fs, fv = PARAMS[MODE][j]
        jid = [names.index(j)]
        robot.write_joint_stiffness_to_sim(k, joint_ids=jid)
        robot.write_joint_damping_to_sim(c, joint_ids=jid)
        robot.write_joint_armature_to_sim(a, joint_ids=jid)
        robot.write_joint_friction_coefficient_to_sim(fs, fs, fv, joint_ids=jid)
    print(f"[{MODE}] 파라미터 적용 완료")

    plot = TrajPlot()

    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID))
    leader.connect()

    follower = None
    if not args.no_follower:
        follower = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID))
        follower.connect()

    def _emergency_stop(signum, frame):
        print("[teleop] 비상 정지 - follower 토크 끄기")
        if follower is not None:
            try:
                follower.bus.disable_torque()
                print("[teleop] 토크 꺼짐")
            except Exception as e:
                print(f"[warn] disable_torque: {e}")
        try:
            leader.disconnect()
        except Exception:
            pass
        try:
            if follower is not None:
                follower.disconnect()
        except Exception:
            pass
        os._exit(0)
    signal.signal(signal.SIGINT, _emergency_stop)

    ld = leader.get_action()
    q = robot.data.default_joint_pos.clone()
    for ix, val in zip(arm_ix, arm_rad(ld)):
        q[0, ix] = val
    q[0, grip_ix] = GRIP_OPEN if grip_open(ld) else GRIP_CLOSE
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    print(f"[teleop] {MODE} 궤적 비교 시작 (Ctrl+C 종료)")

    t0 = time.time()
    step = 0
    try:
        while simulation_app.is_running():
            loop_start = time.perf_counter()

            ld = leader.get_action()
            if follower is not None:
                follower.send_action(ld)
            vals = arm_rad(ld) + [1.0 if grip_open(ld) else -1.0]
            env.step(torch.tensor([vals], dtype=torch.float32, device=env.device))

            if follower is not None and step % READ_EVERY == 0:
                t = time.time() - t0
                for j in PLOT_JOINTS:
                    real_deg = follower.bus.read("Present_Position", j, normalize=True)
                    sim_deg = math.degrees(robot.data.joint_pos[0, plot_ix[j]].item())
                    plot.push(j, t, real_deg, sim_deg)

            if step % VIEW_EVERY == 0:
                plot.update()

            step += 1
            dt = time.perf_counter() - loop_start
            precise_sleep(1.0 / FPS - dt)
    except KeyboardInterrupt:
        print("\n[teleop] 종료")
    finally:
        if follower is not None:
            try:
                follower.bus.disable_torque()
                print("[teleop] follower 토크 꺼짐")
            except Exception as e:
                print(f"[warn] disable_torque: {e}")
        try:
            leader.disconnect()
        except Exception:
            pass
        if follower is not None:
            try:
                follower.disconnect()
            except Exception:
                pass
        try:
            env.close()
        except Exception:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()

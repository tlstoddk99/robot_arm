import time
import signal
import os
import math
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "my_follower"

JOINT = "wrist_flex"

G = 9.81
N_SAMPLES = 30
AMP_DEG = 15.0
CYCLES = 1
PERIOD = 3.0
RATE = 50
SETTLE_SEC = 3.0

CONFIGS = {
    "shoulder_lift": {
        "lever_m": 0.40,
        "pose": {"shoulder_pan": 0.0, "shoulder_lift": 85.0, "elbow_flex": -85.0,
                 "wrist_flex": 15.0, "wrist_roll": 0.0, "gripper": 0.0},
    },
    "elbow_flex": {
        "lever_m": 0.29,
        "pose": {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
                 "wrist_flex": 15.0, "wrist_roll": 0.0, "gripper": 0.0},
    },
    "wrist_flex": {
        "lever_m": 0.15,
        "pose": {"shoulder_pan": 0.0, "shoulder_lift": -30.0, "elbow_flex": -85.0,
                 "wrist_flex": 105.0, "wrist_roll": 0.0, "gripper": 0.0},
    },
}

LEVER_M = CONFIGS[JOINT]["lever_m"]
TARGET_POSE = CONFIGS[JOINT]["pose"]
WOBBLE_JOINTS = [JOINT]

follower = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID))
follower.connect()
follower.bus.enable_torque()

target = {f"{k}.pos": v for k, v in TARGET_POSE.items()}
records = []


def stop(signum=None, frame=None):
    print("\n[정지] 토크 끄기")
    try:
        follower.bus.disable_torque()
    except Exception as e:
        print("disable_torque:", e)
    try:
        follower.disconnect()
    except Exception:
        pass
    if records:
        print("\n=== 측정 요약 ===")
        print(f"{'무게(g)':>8} {'토크(N·m)':>10} {'Load':>8} {'Pos':>8}")
        for g, tq, ld, ps in records:
            print(f"{g:>8.0f} {tq:>10.3f} {ld:>8.1f} {ps:>8.1f}")
    os._exit(0)


signal.signal(signal.SIGINT, stop)


def hold(seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        follower.send_action(target)
        time.sleep(1.0 / RATE)


def wobble():
    steps = int(CYCLES * PERIOD * RATE)
    for i in range(steps + 1):
        offset = AMP_DEG * math.sin(2 * math.pi * (i / RATE) / PERIOD)
        cmd = dict(target)
        for j in WOBBLE_JOINTS:
            cmd[f"{j}.pos"] = TARGET_POSE[j] + offset
        follower.send_action(cmd)
        time.sleep(1.0 / RATE)
    hold(SETTLE_SEC)


def measure():
    loads, positions = [], []
    for _ in range(N_SAMPLES):
        follower.send_action(target)
        loads.append(follower.bus.read("Present_Load", JOINT, normalize=False))
        positions.append(follower.bus.read("Present_Position", JOINT, normalize=False))
        time.sleep(0.02)
    return sum(loads) / len(loads), sum(positions) / len(positions)


def torque_of(grams):
    return (grams / 1000.0) * G * LEVER_M


class Plot:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.set_xlabel("Torque (N·m)")
        self.ax.set_ylabel("Load (raw)")
        self.ax.set_title(f"{JOINT}: Load vs Torque")
        self.ax.grid(True, alpha=0.3)
        self.pts, = self.ax.plot([], [], "o", color="tab:blue", markersize=8)
        self.fit, = self.ax.plot([], [], "-", color="tab:red", alpha=0.7)
        self.fig.tight_layout()

    def update(self):
        if not records:
            return
        tq = [r[1] for r in records]
        ld = [r[2] for r in records]
        self.pts.set_data(tq, ld)
        if len(set(tq)) >= 2:
            n = len(tq)
            sx, sy = sum(tq), sum(ld)
            sxx = sum(x * x for x in tq)
            sxy = sum(x * y for x, y in zip(tq, ld))
            denom = n * sxx - sx * sx
            if denom != 0:
                k = (n * sxy - sx * sy) / denom
                b = (sy - k * sx) / n
                xs = [0, max(tq) * 1.1]
                ys = [k * x + b for x in xs]
                self.fit.set_data(xs, ys)
                self.ax.set_title(f"{JOINT}: Load = {k:.1f}·t + {b:.1f}")
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main():
    print(f"[{JOINT}] 다중 무게 정적 토크 측정")
    print(f"레버암 {LEVER_M*100:.0f}cm, 목표자세 {TARGET_POSE[JOINT]}deg")
    print("무게추 세팅 → 무게(g) 입력 후 Enter → 측정. (Ctrl+C 종료·요약)\n")
    plot = Plot()
    hold(1.5)

    while True:
        s = input("무게(g) 입력 (0=무게추없음): ").strip()
        if not s:
            continue
        try:
            grams = float(s)
        except ValueError:
            print("숫자를 입력하세요.\n")
            continue
        wobble()
        load, pos = measure()
        tq = torque_of(grams)
        records.append((grams, tq, load, pos))
        print(f"    → 무게={grams:.0f}g  토크={tq:.3f}N·m  Load={load:.1f}  Pos={pos:.1f}\n")
        plot.update()


main()

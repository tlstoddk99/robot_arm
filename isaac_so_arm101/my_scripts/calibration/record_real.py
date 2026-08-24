import time
import signal
import os
import math
import csv
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "my_follower"

JOINT = "wrist_flex"

BASE_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": -90.0,
    "gripper": 0.0,
}

RATE = 50
STEP_AMP = 15.0
SIN_AMP = 20.0
SIN_FREQ = 0.2
OUT_DIR = os.path.expanduser("~/robot_arm/isaac_so_arm101/my_scripts/results/traj_data")


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


DURATION = 13 + 3 / SIN_FREQ

follower = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID))
follower.connect()
follower.bus.enable_torque()


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
    os._exit(0)


signal.signal(signal.SIGINT, stop)


def send(offset):
    cmd = {f"{k}.pos": v for k, v in BASE_POSE.items()}
    cmd[f"{JOINT}.pos"] = BASE_POSE[JOINT] + offset
    follower.send_action(cmd)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"real_{JOINT}.csv")

    print(f"[{JOINT}] 실물 궤적 측정 (기본자세로 이동 후 3초 대기)")
    for _ in range(int(3 * RATE)):
        send(0.0)
        time.sleep(1.0 / RATE)

    print(f"측정 시작 (약 {DURATION:.0f}초). Ctrl+C 중단.")
    rows = []
    t0 = time.time()
    while True:
        t = time.time() - t0
        if t > DURATION:
            break
        off = target_offset(t)
        send(off)
        pos_deg = follower.bus.read("Present_Position", JOINT, normalize=True)
        target_deg = BASE_POSE[JOINT] + off
        rows.append((t, target_deg, pos_deg))
        time.sleep(1.0 / RATE)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "target_deg", "present_deg"])
        w.writerows(rows)
    print(f"저장: {path} ({len(rows)} samples)")

    stop()


main()

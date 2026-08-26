import time
import signal
import os
import math
import csv
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "my_follower"

JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]

BASE_POSE = {
    "shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
    "wrist_flex": 0.0, "wrist_roll": -90.0, "gripper": 0.0,
}

# 관절별 위상 (120도씩 차이) — 상호작용 여기
PHASE = {"shoulder_lift": 0.0, "elbow_flex": 2.094, "wrist_flex": 4.189}

RATE = 50
STEP_AMP = 15.0
SIN_AMP = 20.0
SIN_FREQ = 0.2
OUT_DIR = os.path.expanduser("~/robot_arm/isaac_so_arm101/my_scripts/results/traj_data")

DURATION = 13 + 3 / SIN_FREQ


def offset(t, joint):
    ph = PHASE[joint]
    # 계단 구간: 위상에 따라 시차를 둔 계단
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


def send(t):
    cmd = {f"{k}.pos": v for k, v in BASE_POSE.items()}
    for j in JOINTS:
        cmd[f"{j}.pos"] = BASE_POSE[j] + offset(t, j)
    follower.send_action(cmd)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "real_multi.csv")

    print("[multi] 복합 관절 궤적 측정 (기본자세로 이동 후 3초 대기)")
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
        send(t)
        row = [t]
        for j in JOINTS:
            row.append(BASE_POSE[j] + offset(t, j))       # target
            row.append(follower.bus.read("Present_Position", j, normalize=True))  # actual
        rows.append(row)
        time.sleep(1.0 / RATE)

    header = ["t"]
    for j in JOINTS:
        header += [f"{j}_target", f"{j}_pos"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"저장: {path} ({len(rows)} samples)")

    stop()


main()

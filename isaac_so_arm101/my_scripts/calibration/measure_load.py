"""실물 follower: 자세 고정 + Load 측정 (무게추 정적 토크 실험)
   Ctrl+C 시 토크 자동 차단."""
import time, signal, os
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID   = "my_follower"
JOINT = "shoulder_lift"    # 실험 대상
N_SAMPLES = 30             # 평균 샘플 수

f = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID))
f.connect()

def _stop(signum=None, frame=None):
    print("\n[정지] 토크 끄기")
    try: f.bus.disable_torque()
    except Exception as e: print("disable_torque:", e)
    try: f.disconnect()
    except: pass
    os._exit(0)
signal.signal(signal.SIGINT, _stop)

f.bus.enable_torque()
# 현재 자세를 목표로 고정 (팔을 원하는 자세로 두고 시작)
obs = f.get_observation()
target = {k: v for k, v in obs.items() if k.endswith(".pos")}
f.send_action(target)
time.sleep(1.0)

def measure():
    vals, poss = [], []
    for _ in range(N_SAMPLES):
        f.send_action(target)  # 자세 유지
        vals.append(f.bus.read("Present_Load", JOINT, normalize=False))
        poss.append(f.bus.read("Present_Position", JOINT, normalize=False))
        time.sleep(0.02)
    return sum(vals)/len(vals), sum(poss)/len(poss)

print(f"[{JOINT}] 정적 토크 측정. 팔을 수평으로 두고 시작하세요.")
print("각 단계에서 무게추 세팅 후 Enter. (Ctrl+C 종료)\n")
try:
    label = 0
    while True:
        input(f"[측정 {label}] 준비되면 Enter: ")
        load, pos = measure()
        print(f"    → Load(avg)={load:7.1f}   Pos={pos:7.1f}\n")
        label += 1
except KeyboardInterrupt:
    _stop()

import time
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

# leader = 흰색 팔, ttyACM1, 캘리브레이션 id "my_leader"
cfg = SO101LeaderConfig(port="/dev/ttyACM1", id="my_leader")
leader = SO101Leader(cfg)

print("connecting...")
leader.connect()
print("connected. leader 팔을 움직여보세요 (Ctrl+C로 종료)\n")

try:
    while True:
        action = leader.get_action()
        # 관절값 보기 좋게 출력
        print(" | ".join(f"{k}: {v:7.2f}" for k, v in action.items()))
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n종료")
finally:
    leader.disconnect()

"""leader(흰색,ACM1) → follower(검정,ACM0) 텔레오프 (시뮬 없이 실물만)
   follower 재캘리브레이션 후 방향·동작 확인용"""
import time
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

LEADER_PORT   = "/dev/ttyACM1"
LEADER_ID     = "my_leader"
FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID   = "my_follower"

leader   = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID))
follower = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID))

leader.connect()
follower.connect()
print("연결됨. leader를 천천히 움직이세요. (Ctrl+C 종료)")

try:
    while True:
        action = leader.get_action()     # leader 관절 읽기
        follower.send_action(action)      # follower(실물)에 그대로 전달
        time.sleep(0.02)                  # ~50Hz
except KeyboardInterrupt:
    print("\n종료")
finally:
    leader.disconnect()
    follower.disconnect()

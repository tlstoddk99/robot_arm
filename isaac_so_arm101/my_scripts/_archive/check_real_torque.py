"""실물 follower에서 전류/부하/위치 읽기 확인"""
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

f = SO101Follower(SO101FollowerConfig(port="/dev/ttyACM0", id="my_follower"))
f.connect()
print("연결됨. observation 확인:\n")

obs = f.get_observation()
print("=== observation 키 ===")
for k, v in obs.items():
    print(f"  {k}: {v}")

f.disconnect()

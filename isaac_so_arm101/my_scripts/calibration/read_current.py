"""Load 유발 테스트 — 목표를 어긋나게 해서 부하 발생시킴"""
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
import time

f = SO101Follower(SO101FollowerConfig(port="/dev/ttyACM0", id="my_follower"))
f.connect()
f.bus.enable_torque()

m = "shoulder_lift"
# 현재 raw 위치
cur_pos = f.bus.read("Present_Position", m, normalize=False)
print(f"현재 위치(raw): {cur_pos}")

# 목표를 현재에서 살짝 이동 (부하 유발) — 작은 값만!
for delta in [0, 30, 60, 100, -30, -60]:
    target = cur_pos + delta
    f.bus.write("Goal_Position", m, target, normalize=False)
    time.sleep(0.8)  # 이동·안정화
    load = f.bus.read("Present_Load", m, normalize=False)
    cur  = f.bus.read("Present_Current", m, normalize=False)
    pos  = f.bus.read("Present_Position", m, normalize=False)
    print(f"  목표Δ={delta:+4d} → pos={pos:5d}  load={load:6d}  current={cur:6d}")

# 원위치 복귀
f.bus.write("Goal_Position", m, cur_pos, normalize=False)
time.sleep(0.5)
f.disconnect()
print("완료, 원위치 복귀")

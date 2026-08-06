# SO-101 Real2Sim Manipulation Pipeline

실물 **LeRobot SO-101** 로봇 팔과 **NVIDIA Isaac Sim** 디지털 트윈을 연결한 Real-to-Sim(Real2Sim) 매니퓰레이션 파이프라인. 하나의 leader 팔로 실물 follower와 시뮬레이션 로봇을 동시에 텔레오퍼레이션하고, 실물·시뮬 카메라를 나란히 비교해 reality gap을 관찰·정량화한다.

---

## 개요

| 항목 | 내용 |
|------|------|
| 목표 | SO-101 디지털 트윈 구축, 실물↔시뮬 텔레오프, reality gap 관찰/정량화 |
| 하드웨어 | LeRobot SO-101 (leader 흰색 + follower 검정), Innomaker U20CAM-720P ×2 |
| 시뮬 스택 | Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + Python 3.11 + torch 2.7.0(cu128) |
| 로봇 통합 | LeRobot 0.4.4 (isaaclab 환경에 통합 설치) |
| 장면 | 흰 책상 위 RGB(빨강/초록/파랑) 3블록, 듀얼 카메라(탑뷰 + 그리퍼) |

## 시스템 아키텍처

```
        leader 팔 (흰색, /dev/ttyACM*)  ← 사람이 조종
              │  get_action()  (관절 각도, deg)
              ├──────────────────────┬──────────────────────
              ▼                      ▼
     follower.send_action()    시뮬 SO-101 (Isaac Lab)
     (실물 검정 로봇)            offset/부호 변환 후 env.step()
              │                      │
              ▼                      ▼
     실제 카메라 ×2           시뮬 카메라 ×2
     (탑뷰 video2,            (top_cam, gripper_cam)
      그리퍼 video4)                 │
              └──────────┬───────────┘
                         ▼
              matplotlib 2×2 실시간 비교 창
              [SIM top | REAL top / SIM grip | REAL grip]
```

## 주요 구성요소

### 1. 시뮬 장면 (`rgb_blocks_env_cfg.py`)
- RL용 Lift 태스크(`SoArm101LiftCubeEnvCfg`)를 상속해 새 태스크로 확장
- 흰 책상(CuboidCfg) + RGB 블록 3개(빨강=reward 대상 object, 초록/파랑 추가)
- 듀얼 카메라: 탑뷰(거치대 기둥 39cm 위) + 그리퍼(gripper_link 부착)
- 카메라 몸체를 실측 형태로 모델링(판 3.5×3.5×0.5cm + 원통 렌즈 ⌀1.5×1.8cm), 판-렌즈 부모-자식 계층
- 실제 카메라(Innomaker U20CAM-720P, 수평 FOV 102°)에 맞춘 focal_length·화각
- 로봇 색상: URDF 머티리얼을 검정으로 수정(실물과 일치)

### 2. 텔레오프 브릿지 (`teleop_sim.py`)
- leader 팔 관절을 읽어 실물 follower + 시뮬에 동시 전달
- **값 분리**: follower엔 leader 원본, 시뮬엔 offset/부호 변환 적용
- 각도 unwrap(±180 wraparound 보정), 시작 시 leader 자세로 시뮬 초기화
- RL용 action(scale/offset) 우회: `scale=1.0, use_default_offset=False`
- 4개 카메라 뷰 실시간 표시(matplotlib), 저장은 `--save_cam` 옵션

## 실행 방법

### 환경 활성화
```bash
micromamba activate isaaclab
cd ~/robot_arm/isaac_so_arm101
```

### 시뮬 장면 확인 (텔레오프 없이)
```bash
python -m isaac_so_arm101.scripts.zero_agent \
  --task Isaac-SO-ARM101-RGBBlocks-Play-v0 --enable_cameras
```

### 텔레오프 (실물 + 시뮬 + 4카메라)
```bash
# 포트 확인 먼저
ls -l /dev/ttyACM*    # leader/follower 포트 번호 확인 후 teleop_sim.py 상단 상수 반영

python teleop_sim.py              # 실물+시뮬 미러링 + 4뷰 실시간
python teleop_sim.py --save_cam   # + 이미지 저장
python teleop_sim.py --no_follower # 실물 없이 시뮬만 (안전 테스트)
python teleop_sim.py --debug      # 관절값 출력
```

### GUI에서 카메라 뷰 보기
뷰포트 좌상단 `Perspective` → Cameras → `top_cam` / `gripper_cam`

## 하드웨어 매핑

| 장치 | 포트/인덱스 | 캘리브레이션 |
|------|------------|-------------|
| leader (흰색) | `/dev/ttyACM*` | `teleoperators/so_leader/my_leader.json` |
| follower (검정) | `/dev/ttyACM*` | `robots/so_follower/my_follower.json` |
| 탑뷰 카메라 | `/dev/video2` | — |
| 그리퍼 카메라 | `/dev/video4` | — |

> ttyACM 번호는 USB 재연결 시 바뀔 수 있음. 실행 전 `ls -l /dev/ttyACM*`로 확인.

## 관절 매핑 (leader → 시뮬)

- leader `get_action()` → `{관절명.pos: deg}` (범위 대략 ±110~±180°)
- 시뮬 SO-101 관절은 radian → `math.radians(deg - OFFSET) * SIGN`
- Isaac action 순서: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]`(arm 5) + `gripper`(1, Binary)
- 방향은 전부 +1, 영점 offset은 shoulder_lift/elbow_flex에 필요(실측 튜닝)

## 현재 상태 & 다음 단계

**완료**
- [x] Isaac Sim 환경 + RGB 블록 씬 + 듀얼 카메라
- [x] 로봇/카메라 검정 색상 정합
- [x] LeRobot 0.4.4 isaaclab 환경 통합
- [x] leader → 실물+시뮬 동시 텔레오프
- [x] 4카메라 실시간 비교
- [x] 카메라 화각·방향 정합 (눈대중 수준)

**다음 후보**
- [ ] Reality gap 정량화 (SSIM/LPIPS, Task Success Rate, Trajectory RMSE)
- [ ] 정밀 카메라 캘리브레이션 (체커보드 intrinsic/extrinsic, 어안 왜곡 보정)
- [ ] 블록 위치·색 정밀 정합 (실제 이미지 픽셀 측정)
- [ ] 시연 데이터 수집 (관절 + 4카메라 이미지 기록)

# 기술 여정 & 문제 해결 기록

SO-101 Real2Sim 파이프라인을 구축하며 마주친 핵심 난제와 해결 과정. 각 항목은 "무엇이 문제였고, 왜 그랬으며, 어떻게 풀었는지"를 담는다 — 재현과 포트폴리오 설명 양쪽 목적.

---

## 1. 시뮬 환경 안정화 (드라이버·메모리)

**문제**: Isaac Sim 실행 시 torch import 단계에서 `LLVM ERROR: out of memory`, Vulkan `Bus error`, RTX 크래시가 연쇄 발생.

**원인 & 해결** (순서대로 진단):
1. NVIDIA 드라이버 595 회귀 → 580.173으로 다운그레이드, `apt-mark hold`로 고정
2. `libnvidia-gl-580` 손상 → `--reinstall`로 복구 (Vulkan Bus error 해결)
3. **swap 부재로 LLVM OOM** → 32GB swapfile 추가 (`/etc/fstab` 등록)
4. 첫 실행 RTX 셰이더 컴파일이 오래 걸려 "응답 없음" → 강제종료 말고 대기

**교훈**: 시뮬레이터 크래시는 대부분 드라이버·시스템 리소스 문제. 로그 백트레이스를 층별로 읽어 근본 원인을 분리하는 게 핵심.

## 2. RL 리포를 텔레오프용으로 확장

**상황**: 사용한 커뮤니티 리포(`isaac_so_arm101`)는 강화학습 훈련용. 텔레오프는 원래 시나리오에 없었다.

**설계 결정**: 원본을 훼손하지 않고 **상속으로 새 태스크 확장**.
- `SoArm101LiftCubeEnvCfg`를 상속한 `SoArm101RGBBlocksEnvCfg` 신규 작성
- reward가 참조하는 `object`는 유지(빨강 블록), 초록/파랑은 추가 RigidObject
- 태스크 등록(`Isaac-SO-ARM101-RGBBlocks-v0`)

**교훈**: 남의 코드를 확장할 때는 상속·override로 원본 계약(reward 등)을 지키면서 필요한 것만 바꾼다.

## 3. 머티리얼 렌더 크래시

**문제**: 블록에 `PreviewSurfaceCfg`(색 머티리얼)를 넣으니 MDL 렌더 단계에서 `libneuray`/`mdltranslator` 세그폴트.

**해결**: 초기엔 머티리얼을 빼고 회색으로 진행 → 원인 격리 후 재적용. 특정 렌더 타이밍 이슈였고, 이후 안정화됨. (재발 시 `PreviewSurfaceCfg`가 용의자라는 진단 경로 확보)

**교훈**: 크래시가 나면 의심 요소를 하나씩 제거해 최소 재현 케이스를 만든다.

## 4. 카메라 모델링 — 렌즈가 자기 시야를 가림

**문제 1**: 렌즈 원통이 카메라 광축 정면에 놓여 카메라가 렌즈 내부(검은 화면)만 봄.
**해결**: 카메라 near clipping을 0.05로 늘려 렌즈(약 2cm 앞)를 렌더에서 제외. 카메라는 씬을 보고, 렌즈는 외부 시점에서만 보임.

**문제 2**: 렌즈가 판에 파묻힘 / 방향이 시선과 어긋남.
**해결**: 렌즈를 판의 **자식 prim**으로 만들어(`.../top_cam_body/lens`) 판 회전을 자동 상속, 로컬 z앞에 고정. 쿼터니언 수동 계산 제거.

**교훈**: 부모-자식 계층으로 상대 변환을 처리하면 손계산 오류가 사라진다.

## 5. 각도 표현 — 오일러 헬퍼 & wraparound

- 쿼터니언을 손으로 다루기 어려워 **오일러각(deg) 입력 → 쿼터니언 변환 헬퍼**를 만들어 카메라 각도를 직관적으로 조정.
- 텔레오프 중 모터 각도가 ±180°에서 순환(`179→-180`)하며 팔이 순간 튐 → **unwrap 로직**(이전 값 대비 360° 보정)으로 연속화. 실물 연결 시 급격한 모터 명령을 막는 안전장치이기도 함.

## 6. LeRobot ↔ Isaac Lab 통합 (의존성 지옥)

**상황**: leader/follower를 시뮬과 한 프로세스에서 쓰려면 LeRobot을 isaaclab(Python 3.11) 환경에 설치해야 했다.

**장애물과 해결**:
1. 로컬 LeRobot 소스는 Python 3.12 요구 → PyPI **`lerobot==0.4.4`**(3.11 지원)로 우회
2. `pip install lerobot[feetech]`가 **imageio 의존성 충돌**(ResolutionImpossible) → `--no-deps`로 코드만 설치
3. 이후 import 체인을 따라 누락 모듈을 하나씩 추가:
   `deepdiff → orderly_set → cachebox → accelerate → datasets → av → draccus → mergedeep → typing_inspect → scservo_sdk(feetech-servo-sdk)`
4. **핵심 제약 지키기**: `numpy==1.26.0` 고정(Isaac 필수, numpy 2.x는 ABI 파괴), `packaging 23.0` 유지(isaacsim-kernel 고정)

**검증**: 매 설치 후 `--dry-run`으로 numpy/torch/isaacsim이 안 바뀌는지 확인. 최종적으로 torch 2.7.0 + numpy 1.26.0 유지, Isaac 시뮬 정상, feetech·leader·follower import 성공.

**교훈**: 취약한 환경에 무거운 패키지를 얹을 때는 `--dry-run`으로 영향 범위를 먼저 보고, `--no-deps` + 선별 설치로 핵심 의존성(numpy 등)을 보호한다. 이것이 이 프로젝트에서 가장 까다롭고 값진 문제 해결이었다.

## 7. Action 파이프라인 왜곡 (RL vs 텔레오프)

**문제**: leader 각도를 넣어도 시뮬 팔의 방향·범위가 기대와 달랐다.

**원인**: RL용 `JointPositionActionCfg`가 `scale=0.5, use_default_offset=True`.
정책 신경망 학습을 위해 값을 "기본자세 + 정규화출력×0.5"로 변형하는 설정인데, 텔레오프(leader 각도 = 목표각)에는 방해가 됨.

**해결**: 텔레오프 스크립트에서 `scale=1.0, use_default_offset=False`로 override. gripper는 `BinaryJointPositionActionCfg`(열림/닫힘 2상태)라 부호로 여닫도록 별도 처리.

**교훈**: 같은 프레임워크라도 용도(학습 vs 텔레오프)에 따라 action 의미가 다르다. 원본을 이해하고 목적에 맞게 재해석한다.

## 8. 영점 정합 — 절대 미러링

**요구**: 시뮬을 leader 자세에 맞추기(사람이 시뮬을 흉내 내지 않도록).

**해결**:
- 시작 시 leader 자세를 읽어 `write_joint_state_to_sim`으로 시뮬을 그 자세로 초기화 → 시작 점프 제거
- 관절별 offset(shoulder_lift +110°, elbow_flex −90°, wrist_roll −45° 등)을 실측 튜닝해 leader↔시뮬 자세 일치
- 시뮬 기본자세·관절순서는 하드코딩 대신 런타임에 `robot.data.default_joint_pos`, `joint_names`로 읽어 자동 매핑

## 9. 실물 follower 안전 통합

- 실물이 움직이므로 **단계 분리**: (A) leader→follower 단독 테스트로 안전 확인 → (B) 시뮬+실물 통합
- follower엔 leader **원본 값** 전달(LeRobot이 같은 SO-101 기구학으로 자동 매핑), 시뮬용 offset과 분리
- 시작 전 공간 확보·비상정지(Ctrl+C) 준비

## 10. 카메라 정합 (intrinsic)

- 실제 카메라 모델 식별: `lsusb`/`v4l2-ctl` → **Innomaker U20CAM-720P** (수평 FOV 102°)
- FOV↔focal_length 관계로 시뮬 `focal_length`·`horizontal_aperture` 설정
- 어안 왜곡 때문에 "전체 화각 매칭"과 "중앙 물체 크기 매칭"이 다른 focal 값을 요구 → 용도에 맞게 눈대중 튜닝, 정밀 왜곡 보정은 데이터 수집 단계로 유보(premature optimization 회피)

---

## 회고 요약

이 프로젝트의 가치는 "완성된 데모"보다 **엔지니어링 판단의 연속**에 있다:
- 크래시를 층별로 진단해 근본 원인 분리 (드라이버/메모리/렌더)
- 취약한 환경을 보호하며 무거운 의존성 통합 (dry-run + no-deps + 핵심 핀 유지)
- 프레임워크의 원래 의도(RL)를 이해하고 새 목적(텔레오프)으로 재해석
- 안전을 위한 단계적 통합(시뮬 검증 → 실물 연결)
- 완벽주의 대신 단계에 맞는 정밀도 선택(카메라 정합 수준 판단)

end-to-end Real2Sim 매니퓰레이션 파이프라인(디지털 트윈, 실물↔시뮬 텔레오프, 카메라 정합, reality gap 관찰)을 실제로 구축했으며, 다른 로봇·태스크로 일반화 가능한 방법론을 확보했다.

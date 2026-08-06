# LeRobot 0.4.4 → isaaclab 환경 통합 (재현 레시피)

Isaac Sim 5.1 / Python 3.11 / torch 2.7.0(cu128) / numpy 1.26.0 환경을 **깨지 않고** LeRobot을 설치하는 검증된 순서. 핵심은 `numpy==1.26.0` 고정과 `--no-deps` 선별 설치.

## 왜 까다로운가
- 로컬 LeRobot 소스는 Python 3.12 요구 → PyPI 0.4.4(3.11 지원) 사용
- `pip install lerobot[feetech]`는 imageio 의존성 충돌(ResolutionImpossible)
- numpy 2.x로 업그레이드하려 함 → Isaac이 numpy 1.26 ABI에 의존하므로 **금지**
- packaging은 isaacsim-kernel이 23.0으로 고정 → 유지

## 설치 순서

```bash
micromamba activate isaaclab

# 0) 매 단계 dry-run으로 numpy/torch/isaac 안 건드리는지 확인하는 습관
#    pip install <pkg> --dry-run 2>&1 | grep -iE "numpy|torch|isaac|would install"

# 1) lerobot 코드만 (의존성 충돌 회피)
pip install "lerobot[feetech]==0.4.4" --no-deps
pip install pyserial "numpy==1.26.0"

# 2) import 체인 따라 누락 모듈 선별 추가 (전부 numpy 보호)
pip install deepdiff orderly-set cachebox --no-deps    # deepdiff 계열
pip install accelerate --no-deps                        # lerobot.utils
pip install datasets "numpy==1.26.0"                    # dill/multiprocess/pandas/pyarrow/xxhash 동반
pip install av --no-deps                                # 영상 코덱
pip install draccus mergedeep pyyaml-include pfzy --no-deps           # 설정(teleoperators)
pip install typing_inspect typing_extensions mypy_extensions --no-deps
pip install "feetech-servo-sdk>=1.0.0,<2.0.0" --no-deps # → import scservo_sdk

# 3) 검증
python -c "from lerobot.motors.feetech import FeetechMotorsBus; print('feetech OK')"
python -c "from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig; print('leader OK')"
python -c "from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig; print('follower OK')"
python -c "import torch,numpy; print(torch.__version__, numpy.__version__)"  # 2.7.0+cu128 / 1.26.0
```

## 지켜야 할 불변식
- `numpy == 1.26.0` (Isaac 필수, 2.x 금지)
- `packaging == 23.0` (isaacsim-kernel 고정 — 올리지 말 것)
- `torch 2.7.0+cu128`, `torchvision 0.22.0` (건드리지 말 것)

## API 요점 (0.4.4)
- 경로는 `so_leader`/`so_follower` (NOT `so101_leader`)
- 클래스: `SO101Leader`/`SO101LeaderConfig`, `SO101Follower`/`SO101FollowerConfig`
- leader: `connect()` → `get_action()`(→ `{관절.pos: deg}`) → `disconnect()`
- follower: `connect()` → `send_action(action_dict)` → `disconnect()`
- 캘리브레이션 id로 자동 로드: leader `my_leader`, follower `my_follower`

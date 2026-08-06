#!/bin/bash
# SO-101 Real2Sim 프로젝트 백업 스크립트
# 핵심 코드 + 환경 정보 + 캘리브레이션을 타임스탬프 폴더로 저장한다.
set -e

ROOT=~/robot_arm/isaac_so_arm101
STAMP=$(date +%Y%m%d_%H%M%S)
DEST=~/robot_arm/backups/$STAMP
mkdir -p "$DEST"

echo "[backup] → $DEST"

# 1. 핵심 소스 파일
TASK_DIR=src/isaac_so_arm101/tasks/lift
ROBOT_DIR=src/isaac_so_arm101/robots/trs_so101

mkdir -p "$DEST/task" "$DEST/robot" "$DEST/scripts"
cp "$ROOT/$TASK_DIR/rgb_blocks_env_cfg.py"        "$DEST/task/"        2>/dev/null || true
cp "$ROOT/$TASK_DIR/__init__.py"                  "$DEST/task/"        2>/dev/null || true
cp "$ROOT/$TASK_DIR/joint_pos_env_cfg.py"         "$DEST/task/"        2>/dev/null || true
cp "$ROOT/$TASK_DIR/lift_env_cfg.py"              "$DEST/task/"        2>/dev/null || true
cp "$ROOT/$ROBOT_DIR/so_arm101.py"                "$DEST/robot/"       2>/dev/null || true
cp "$ROOT/$ROBOT_DIR/urdf/so_arm101.urdf"         "$DEST/robot/"       2>/dev/null || true
cp "$ROOT/teleop_sim.py"                          "$DEST/scripts/"     2>/dev/null || true
cp "$ROOT/test_leader.py"                         "$DEST/scripts/"     2>/dev/null || true
cp "$ROOT/test_follower.py"                       "$DEST/scripts/"     2>/dev/null || true
cp "$ROOT/test_real_cam.py"                       "$DEST/scripts/"     2>/dev/null || true

# 2. 캘리브레이션 (leader/follower)
mkdir -p "$DEST/calibration"
cp -r ~/.cache/huggingface/lerobot/calibration/* "$DEST/calibration/" 2>/dev/null || true

# 3. 환경 스냅샷 (재현용)
micromamba run -n isaaclab pip freeze > "$DEST/pip_freeze.txt" 2>/dev/null || true
micromamba run -n isaaclab python -c "import torch,numpy; print('torch',torch.__version__); print('numpy',numpy.__version__)" > "$DEST/versions.txt" 2>/dev/null || true
nvidia-smi --query-gpu=name,driver_version --format=csv > "$DEST/gpu.txt" 2>/dev/null || true

# 4. 문서 (있으면)
cp ~/robot_arm/README.md  "$DEST/" 2>/dev/null || true
cp ~/robot_arm/JOURNEY.md "$DEST/" 2>/dev/null || true

echo "[backup] 완료. 파일 목록:"
find "$DEST" -type f | sed "s|$DEST/||"

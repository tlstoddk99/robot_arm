# Install on macOS (Runtime Profile, No Docker)

This guide installs the runtime workflow directly on macOS (no Docker), adapted from the Linux `Dockerfile.runtime` profile.

Scope: data collection + inference/deploy scripts.  
Not included: full dev/train Docker toolchain.

## 1) Supported macOS setup

- macOS 13+ recommended
- Homebrew installed
- Python 3.12
- Works on Apple Silicon (M1/M2/M3) and Intel Macs

## 2) Install system prerequisites (Homebrew)

```bash
brew update
brew install \
  python@3.12 \
  git \
  git-lfs \
  ffmpeg \
  glfw \
  cmake
```

Initialize Git LFS once:

```bash
git lfs install
```

## 3) Create and activate a virtual environment

From repo root (`lerobot_mujoco_sim/`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 4) Install runtime Python dependencies

Unlike Linux runtime Docker, macOS should use default PyPI index for PyTorch.

```bash
python -m pip install \
  torch==2.7.0 \
  torchvision==0.22.0
```

```bash
python -m pip install \
  mujoco==3.1.6 \
  numpy \
  Pillow \
  glfw \
  termcolor \
  PyYAML \
  mink
```

Install LeRobot (same pin as `Dockerfile.runtime`; dataset format v3.0):

```bash
python -m pip install "lerobot==0.5.1"
```

## 5) Configure MuJoCo rendering on macOS

Use GLFW on macOS (not EGL/OSMesa):

```bash
export MUJOCO_GL=glfw
```

To persist it:

```bash
echo 'export MUJOCO_GL=glfw' >> ~/.zshrc
```

## 6) Optional Hugging Face CLI

If you need dataset upload/login:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Then open a new shell and login:

```bash
huggingface-cli login
```

## 7) Project asset prep

If the archive exists, unpack:

```bash
if [ -f asset/objaverse/plate_11.zip ]; then
  unzip -o asset/objaverse/plate_11.zip -d asset/objaverse/
fi
```

## 8) Verify installation

```bash
python -c "import mujoco, torch, lerobot; print('ok')"
python scripts/collect/manual_collect_data.py --help
python scripts/deploy/deploy_act.py --help
```

## 9) Typical runtime usage

Collect demos:

```bash
export MUJOCO_GL=glfw
python scripts/collect/manual_collect_data.py \
  --env-robot-profile so101 \
  --num-demo 20 \
  --repo-name <your-dataset-repo> \
  --root data/demo_data_so101 \
  --offline-local-only
```

Deploy ACT:

```bash
python scripts/deploy/deploy_act.py --config configs/deploy_act.yaml
```

## Troubleshooting (macOS)

- If `python3.12` is not found, run `brew info python@3.12` and add brew Python to `PATH`.
- If MuJoCo context creation fails, confirm `MUJOCO_GL=glfw` in the same shell session.
- If `opencv-python` wheel issues appear on older Macs, try:
  - `python -m pip install opencv-python-headless`
- If `lerobot` install fails due to Linux-only dependencies (for example around `evdev`), this is an upstream packaging compatibility issue; prefer Linux runtime for strict parity with Docker.
- On Apple Silicon, keep all installs inside one arm64 shell/venv (`uname -m` should be `arm64`) to avoid mixed-arch wheel issues.


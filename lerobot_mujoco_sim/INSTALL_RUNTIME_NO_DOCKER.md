# Install Without Docker (Runtime Profile)

This guide installs the project directly on your host machine, matching the `Dockerfile.runtime` setup (capture + inference, no Jupyter/dev stack).

## 1) Host requirements

- OS: Ubuntu 24.04 (recommended, same as runtime image base)
- Python: 3.12
- Optional GPU for EGL rendering/inference acceleration
- Git + internet access for Python dependencies and LeRobot install

If you are on a different distro/version, package names may differ.

## 2) Install system packages

From `lerobot_mujoco_sim/` (or anywhere), run:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  git-lfs \
  build-essential \
  python3.12 \
  python3.12-venv \
  python3.12-dev \
  python3-pip \
  ffmpeg \
  unzip \
  libgl1 \
  libglib2.0-0 \
  libglfw3 \
  libglew2.2 \
  libosmesa6 \
  libegl1 \
  libx11-6 \
  libxext6 \
  libxrender1 \
  libxrandr2 \
  libxinerama1 \
  libxi6 \
  libxcursor1 \
  libxkbcommon0 \
  libsm6 \
  libxmu6
```

Initialize Git LFS once:

```bash
git lfs install
```

## 3) Create and activate virtual environment

In repo root (`lerobot_mujoco_sim/`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 4) Install runtime Python dependencies

Install PyTorch CPU wheels (same as runtime profile):

```bash
python -m pip install \
  torch==2.7.0 \
  torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

Install MuJoCo + runtime libs:

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

## 5) (Optional) Install Hugging Face CLI

For login/upload workflows:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Then start a new shell (or `source ~/.bashrc`) and run:

```bash
huggingface-cli login
```

## 6) Project-specific setup

If the object asset archive exists, unpack it:

```bash
if [ -f asset/objaverse/plate_11.zip ]; then
  unzip -o asset/objaverse/plate_11.zip -d asset/objaverse/
fi
```

Set runtime env defaults:

```bash
export MUJOCO_GL=egl
```

If you use an NVIDIA GPU and EGL, make sure host GPU drivers are installed and working.

## 7) Smoke tests

Check imports:

```bash
python -c "import mujoco, torch, lerobot; print('ok')"
```

Run one of the project scripts (examples):

```bash
python scripts/collect/manual_collect_data.py --help
python scripts/deploy/deploy_act.py --help
```

## 8) Typical runtime usage (no Docker)

From repo root with venv activated:

```bash
export MUJOCO_GL=egl
python scripts/collect/manual_collect_data.py \
  --env-robot-profile so101 \
  --num-demo 20 \
  --repo-name <your-dataset-repo> \
  --root data/demo_data_so101 \
  --offline-local-only
```

Deploy a trained ACT checkpoint:

```bash
python scripts/deploy/deploy_act.py --config configs/deploy_act.yaml
```

## Troubleshooting

- `ModuleNotFoundError`: ensure your venv is activated (`which python` should point to `.venv`).
- MuJoCo render/context errors: confirm `MUJOCO_GL=egl`; try `MUJOCO_GL=osmesa` as fallback for headless CPU rendering.
- Build errors for `evdev`/native wheels: ensure `build-essential` and `python3.12-dev` are installed.
- `huggingface-cli` not found: restart shell or add `~/.local/bin` to your `PATH`.


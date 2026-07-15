# LeRobot + MuJoCo: ACT Pipeline

> **Original Work Credit:**  
> This repository is based on [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial) by [Jeongeun Park](https://github.com/jeongeun980906). This fork focuses on the **ACT (Action-Chunking Transformer)** pipeline for the EE5108 mini-project.

Collect demonstration data in MuJoCo, train an ACT policy, and deploy it in simulation. Single-task pick-and-place (SO-101 arm, blue block → bin).

LeRobot SO-101 arm simulation in Mujoco

## Table of Contents

- [Installation](#installation)
- [EE5108 Mini-Project Workflow](#ee5108-mini-project-workflow)
- [Project Structure](#project-structure)
- [Collect Data](#collect-data)
- [Convert dataset from v2.1 to v3.0](#convert-dataset-from-v21-to-v30)
  - [Local conversion (disk only)](#local-conversion-disk-only)
  - [Hugging Face Hub: convert and push](#hugging-face-hub-convert-and-push)
- [Playback Data](#playback-data)
- [Train ACT](#train-act)
- [Deploy ACT](#deploy-act)
- [Arm Profiles & Config](#arm-profiles--config)
- [Upload to Hugging Face](#upload-to-hugging-face)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Installation

Python 3.12 recommended. Docker is the simplest option.

### Docker (recommended)

Two images:

- **Runtime** (`Dockerfile.runtime`): data collection + policy deployment (smaller, no Jupyter).
- **Train/dev** (`Dockerfile`): same as runtime + Jupyter for local training.

Build from `lerobot_mujoco_sim/`:

```bash
docker build -t lerobot_mujoco_sim:runtime -f Dockerfile.runtime .
docker build -t lerobot_mujoco_sim:train -f Dockerfile .
```

Run (GPU):

```bash
docker run --rm -it --gpus all -v "$PWD:/workspace" lerobot_mujoco_sim:runtime
```

Inside the container: `cd /workspace/lerobot_mujoco_sim` (or `/workspace` if you mounted the repo root).

With docker-compose:

```bash
docker compose up -d lerobot-runtime
docker compose exec lerobot-runtime bash
cd /workspace/lerobot_mujoco_sim
```

## SO-101 / SO-100 Arm Assets

- Vendored: `third_party/SO-ARM100`
- MuJoCo assets: `asset/so_arm100/SO101/`, `asset/so_arm100/SO100/`
- Scene used for ACT: `asset/scene_so101_y.xml` (SO-101, blue block → bin)

## EE5108 Mini-Project Workflow

End-to-end flow: capture data → upload → train ACT (e.g. Colab) → deploy in MuJoCo.

### 1) Build and start runtime

From repo root on host:

```bash
docker build -t lerobot_mujoco_sim:runtime -f Dockerfile.runtime .
docker run --rm -it --gpus all -v "$PWD:/workspace" lerobot_mujoco_sim:runtime
```

Then inside container: `cd /workspace/lerobot_mujoco_sim`.

### 2) Capture data (SO-101)

**Manual teleop** (recommended first time):

```bash
python scripts/collect/manual_collect_data.py \
  --env-robot-profile so101 \
  --num-demo 20 \
  --repo-name <your-dataset-repo> \
  --root data/demo_data_so101 \
  --offline-local-only
```

**Scripted batch** (Mink FSM, no teleop):

```bash
python scripts/collect/batch_collect_data.py \
  --env-robot-profile so101 \
  --num-demo 200 \
  --repo-name <your-dataset-repo> \
  --root data/demo_data_so101 \
  --offline-local-only
```

Both write a LeRobot-style dataset under `data/demo_data_so101` (or a `_fresh_*` variant if the directory already exists and `--offline-local-only` is set).

Docker and local install targets **LeRobot 0.5.x**, so new datasets use **`codebase_version` v3.0**. With that API, `LeRobotDataset.add_frame` takes only a frame dictionary and expects a **`task`** field each step (the collectors pass your `task_name` from config). Call **`dataset.finalize()`** after recording so v3 parquet/metadata is flushed; the collector scripts do this on exit, and the collect notebook closes with `finalize()` before shutdown.

If you have an **older on-disk dataset** whose `meta/info.json` still says **`v2.1`**, convert it before training or deploying with the current stack. See [Convert dataset from v2.1 to v3.0](#convert-dataset-from-v21-to-v30).

### 3) Upload dataset to Hugging Face

You'll need a Huggingface account and an access token to create a dataset.
Login (one of):

```bash
python -c "from huggingface_hub import login; login(token='hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')"
```

Upload:

```bash
python scripts/hf/upload_hf.py \
  --folder data/demo_data_so101 \
  --repo-id <your-hf-username>/<your-dataset-repo> \
  --repo-type dataset \
  --create-if-missing \
  --message "Upload SO101 dataset from demo_data_so101"
```

Use `--private` for a private repo.

### 4) Train ACT

- Use Google Colab (recommended) and run the notebook `notebooks/EE5108_training_act.ipynb`.
- In the notebook, set `dataset.repo_id` to `<your-hf-username>/<your-dataset-repo>` (the dataset you uploaded in step 3).
- The notebook trains an ACT checkpoint and writes `deploy_metadata.json` into the checkpoint directory for easy deployment.

### 5) Deploy in MuJoCo

Copy the trained checkpoint (e.g. into `checkpoints/act_y/`), then:

```bash
python scripts/deploy/deploy_act.py --checkpoint checkpoints/act_y
```

Config-file driven deploy (recommended):

```bash
python scripts/deploy/deploy_act.py --config configs/deploy_act.yaml
```

CPU-only:

```bash
python scripts/deploy/deploy_act.py --checkpoint checkpoints/act_y --device cpu
```

## Project Structure

```
lerobot_mujoco_sim/
├── README.md
├── Dockerfile              # Train/dev image (Jupyter + deps)
├── Dockerfile.runtime      # Runtime image (collect + deploy)
├── docker-compose.yml
├── asset/
│   ├── scene_so101_y.xml   # SO-101 pick-and-place scene
│   ├── so_arm100/          # SO-101/SO-100 MuJoCo assets
│   └── tabletop/
├── configs/
│   ├── collect_batch.yaml
│   ├── collect_manual.yaml
│   └── collect_data.yaml
├── scripts/
│   ├── collect/            # manual_collect_data.py, batch_collect_data.py
│   ├── deploy/             # deploy_act.py
│   ├── train/              # train_act.py
│   ├── visualize/          # visualize_data.py
│   ├── convert_lerobot_dataset_v21_to_v30_local.py  # v2.1 → v3.0 migration wrapper
│   └── hf/                 # upload_hf.py
├── mujoco_env/             # MuJoCo environment
├── controllers/            # Mink FSM for batch collection
├── so101/                  # SO-101 kinematics
└── third_party/SO-ARM100/
```

Data and checkpoints are local (e.g. `data/`, `checkpoints/`) and typically not committed.

## Collect Data

Default task: pick blue block, place in bin. Scene: `asset/scene_so101_y.xml`.

### Manual collection (keyboard)

- **WASD** + **R/F** + arrows: joint control (see [Keyboard controls](#keyboard-controls-so-101)).
- **SPACE**: toggle gripper. **Z**: reset and discard current episode.
- Recording starts on first movement; finish a successful pick-and-place to save the episode.

Config and YAML:

```bash
python scripts/collect/manual_collect_data.py --config configs/collect_manual.yaml
python scripts/collect/batch_collect_data.py --config configs/collect_batch.yaml --num-demo 100
```

### Keyboard controls (SO-101)


| Key   | Action                  |
| ----- | ----------------------- |
| A / D | Shoulder pan            |
| W / S | Shoulder lift           |
| R / F | Elbow flex              |
| ↑ / ↓ | Wrist flex              |
| ← / → | Wrist roll              |
| SPACE | Toggle gripper          |
| Z     | Reset & discard episode |


### Scripted batch

`batch_collect_data.py` uses a Mink-based FSM to generate demonstrations without teleop. Same dataset format as manual collection.

## Convert dataset from v2.1 to v3.0

Use this when `meta/info.json` still says **`"codebase_version": "v2.1"`** (older LeRobot / on-disk layout) and you want **LeRobot 0.5.x** and the **v3.0** dataset format used by this repo.

**What runs under the hood:** [Hugging Face LeRobot](https://github.com/huggingface/lerobot) `convert_dataset` (`lerobot.scripts.convert_dataset_v21_to_v30`). This repository ships **`scripts/convert_lerobot_dataset_v21_to_v30_local.py`**, a wrapper that:

- Runs the same conversion as upstream.
- **Repairs `spawn.block_xyz` in episode parquet** when it is stored as a NumPy “object vector” of four `(3,)` arrays (that layout breaks `pyarrow.Schema.from_pandas` during sharding). Cells are rewritten to nested lists before conversion. Skip with `--no-repair-parquet` if you know your parquet is already safe.

You can use the tool in two ways: **only on disk** (default), or **disk + upload to the Hugging Face Hub**.

### Prerequisites on disk (typical v2.1 tree)

- `meta/info.json` with `codebase_version` **v2.1**
- `meta/episodes.jsonl`, `meta/tasks.jsonl`, `meta/episodes_stats.jsonl`
- `data/` (and `videos/` if the run used encoded video)

`meta/stats.json` is **not** required (global stats are recomputed from `episodes_stats.jsonl`).

### `--root` vs `--repo-id` (read this once)

| Argument | Meaning |
|----------|--------|
| **`--root`** | **Filesystem path** to the dataset: the folder that directly contains `meta/` and `data/` (not the git repo root unless your dataset lives there). |
| **`--repo-id`** | **Not a path.** It is the Hugging Face **dataset repo id** (`user-or-org/dataset-name`) when pushing, or a **placeholder** such as `local/my_run` for local-only conversion (default: `local/<folder_name>` if omitted). |

**Docker:** If you mount this project at `/workspace`, the dataset is usually `/workspace/data/<dataset_name>`, **not** `/workspace/lerobot_mujoco_sim/data/...`, unless you explicitly mounted that nested path.

### Environment

Use **Python 3.12** with **`lerobot==0.5.1`** (or another 0.5.x that includes the v2.1→v3.0 script): venv on the host, this repo’s image if it already has `lerobot`, or the Docker one-liner below.

### Local conversion (disk only)

Goal: turn the folder at `--root` into a **v3.0** dataset **on your machine**. Nothing is uploaded; **no Hugging Face login** is required.

From the repo root:

```bash
pip install "lerobot==0.5.1"   # in a venv if your system Python is PEP 668–managed

python scripts/convert_lerobot_dataset_v21_to_v30_local.py \
  --root /absolute/or/relative/path/to/my_v21_dataset \
  --repo-id local/my_v21_dataset
```

If you omit `--repo-id`, it defaults to `local/<folder_name>`.

**What happens to files:** upstream writes a v3.0 tree beside your dataset, moves the original v2.1 tree to **`{dataset_name}_old`** (next to `--root`), then moves the converted tree **into** the original `--root` path. Your training path stays the same; only the format changes.

**Check:**

```bash
python -c "import json; print(json.load(open(\"path/to/my_v21_dataset/meta/info.json\"))[\"codebase_version\"])"
# expect: v3.0
```

Optional: load with `LeRobotDataset` and read one step to confirm.

### Hugging Face Hub: convert and push

Goal: same **in-place** conversion on disk, then **upload the v3.0 dataset** to a Hub **dataset** repository so others (or you, from Colab) can load it with `repo_id=user/dataset`.

**Before you run**

1. **Create an empty dataset repo** on the Hub (for example [New dataset](https://huggingface.co/new-dataset)), e.g. `myuser/so101_pick_place_v30`. You need **write** access.
2. **Authenticate:** `huggingface-cli login`, or export `HF_TOKEN` with permission to push to that repo.

**Command** (real Hub id + `--push-to-hub`):

```bash
python scripts/convert_lerobot_dataset_v21_to_v30_local.py \
  --root /path/to/my_v21_dataset \
  --repo-id myuser/so101_pick_place_v30 \
  --push-to-hub
```

Optional: `--branch main` (or another branch) if your workflow uses non-default branches.

**Important:** Conversion **still rewrites the folder at `--root`** (original v2.1 is moved to `*_old` next to it). If you must **keep the v2.1 tree unchanged** at the original path, **copy** the dataset to another directory and pass that copy as `--root`.

**Same thing with upstream only** (no `spawn.block_xyz` repair from this repo):

```bash
python -m lerobot.scripts.convert_dataset_v21_to_v30 \
  --repo-id=myuser/so101_pick_place_v30 \
  --root=/absolute/path/to/my_v21_dataset \
  --push-to-hub=true
```

If that module path is missing, install `lerobot` from PyPI. Prefer this repo’s wrapper for datasets collected here, so parquet repair runs automatically.

For a dataset that is **already v3.0 on disk** and you only want to upload it, see [Upload to Hugging Face](#upload-to-hugging-face) (`scripts/hf/upload_hf.py`).

### Docker (one-off, local conversion example)

```bash
docker run --rm -it \
  -v /path/to/lerobot_mujoco_sim:/workspace \
  -w /workspace \
  python:3.12-bookworm bash -lc '
    pip install -q "lerobot==0.5.1" &&
    python scripts/convert_lerobot_dataset_v21_to_v30_local.py \
      --root /workspace/data/my_v21_dataset \
      --repo-id local/my_v21_dataset
  '
```

Add Hugging Face credentials inside the container (`huggingface-cli login` or `-e HF_TOKEN=...`) and append `--push-to-hub --repo-id youruser/your-dataset` for a Hub push.

### Wrapper flags (quick reference)

- **`--push-to-hub`** — after conversion, push to the dataset repo named by `--repo-id`.
- **`--branch`** — Hub branch (upstream default: `main`).
- **`--data-file-size-in-mb`**, **`--video-file-size-in-mb`** — sharding for parquet / video.
- **`--force-conversion`** — skip upstream guard if a Hub v3 snapshot already exists (use carefully).
- **`--no-repair-parquet`** — do not rewrite `spawn.block_xyz` in parquet.
- **`--skip-preflight`**, **`--no-progress`**, **`--no-logging-setup`** — advanced / CI.

### Caveats

- Unusual dataset layouts can hit edge cases in older LeRobot versions; use **`lerobot==0.5.1`** when possible.
- **v2.0 → v3.0** is often not a single hop; migrate to v2.1 with older tooling first if needed ([LeRobot discussions](https://github.com/huggingface/lerobot)).

## Playback Data

Replay saved episodes in MuJoCo:

```bash
python scripts/visualize/visualize_data.py
```

Or use the notebook `notebooks/2.visualize_data.ipynb`.

## Train ACT

Recommended (students): run the Colab notebook `notebooks/EE5108_training_act.ipynb`.

**Hub tokens:** Do not paste Hugging Face access tokens into notebook source if the notebook might be shared or checked into git. Prefer `HF_TOKEN` (or Colab Secrets), `huggingface-cli login` on your machine, or the `token.txt` pattern described under [Deploy ACT](#deploy-act). The training notebook reads credentials from the environment.

Minimal local training example (if you already have the environment set up):

```bash
python scripts/train/train_act.py
```

Checkpoint is written to `checkpoints/act_y/` (including `deploy_metadata.json` for deployment). For full training (e.g. Colab), use the LeRobot pipeline with your Hugging Face dataset.

## Deploy ACT

Run the policy in the same SO-101 MuJoCo scene using a trained ACT checkpoint.

```bash
python scripts/deploy/deploy_act.py --checkpoint checkpoints/act_y
```

`scripts/deploy/deploy_act.py` builds the ACT policy from your checkpoint’s `config.json`, then loads dataset normalization stats (needed to denormalize the action outputs correctly).

The dataset stats come from one of these places:

- If `checkpoints/<your-checkpoint>/deploy_metadata.json` exists, `deploy_act.py` uses that directly.
- Otherwise, it looks for local dataset metadata under `--dataset-root`.
- It expects `meta/info.json` (includes `features`).
- It expects `meta/stats.json` (normalization statistics).
- If those local files are missing, it downloads only `meta/info.json` and `meta/stats.json` from the Hugging Face **dataset** repo given by `--dataset-repo-id`.

Notes / gotchas:

- This repo’s deployment script is **MuJoCo sim-only**. Do not use `lerobot-record` for this workflow (it targets the physical robot and requires a hardware `--robot.port`).
- Make sure `--dataset-repo-id` points to a Hugging Face *dataset* repo (uploaded with `--repo-type dataset`), not a model/policy repo.

Common command (local dataset stats):

```bash
python scripts/deploy/deploy_act.py \
  --checkpoint checkpoints/act_y \
  --dataset-root data/demo_data_so101
```

### Using the YAML config

Spawn bounds and runtime defaults for deploy live in `configs/deploy_act.yaml`.

```bash
python scripts/deploy/deploy_act.py --config configs/deploy_act.yaml
```

CLI flags still override YAML values. Options include: `--xml-path`, `--device cpu`, `--seed`, `--spawn-x-min`, etc. See `--help`.

### Important flags (most used)

Run `python scripts/deploy/deploy_act.py --help` for the full list, but these are the key ones:

- `--checkpoint`: checkpoint folder containing at least `config.json` (and optionally `deploy_metadata.json`)
- `--dataset-root`: local dataset folder (default is set in the YAML). Expected to contain `meta/info.json` and `meta/stats.json`.
- `--dataset-repo-id`: Hugging Face dataset repo id used only if local `meta/` files aren’t present
- `--xml-path`: MuJoCo scene XML

### If your dataset is private

Set `HF_TOKEN` in the environment, or (when using the provided docker-compose runtime) place a token in `token.txt` at the repo root so the container can export it.

## Arm Profiles & Config

Profiles: `so101` (default), `so100`, `omy`. Defaults come from `configs/collect_data.yaml` (e.g. `env_robot_profile: so101`, `xml_path: ./asset/scene_so101_y.xml`). Override at runtime:

```bash
python scripts/collect/manual_collect_data.py --env-robot-profile so100
```

Spawn bounds and scene settings are in the same config.

## Upload to Hugging Face

After collecting data in **v3.0** layout (or once you have converted v2.1 → v3.0; see [Convert dataset from v2.1 to v3.0](#convert-dataset-from-v21-to-v30)):

```bash
huggingface-cli login   # or set HF_TOKEN
python scripts/hf/upload_hf.py \
  --folder data/demo_data_so101 \
  --repo-id <your-hf-username>/<your-dataset-repo> \
  --repo-type dataset \
  --create-if-missing
```

If your on-disk dataset is still **v2.1**, use the conversion section’s **Hugging Face Hub** flow (`--push-to-hub` with `convert_lerobot_dataset_v21_to_v30_local.py`) instead of uploading raw v2.1 with this script, unless you intentionally want the legacy format on the Hub.

If using docker-compose with `HF_HUB_OFFLINE=1`, unset it for uploads.

## Acknowledgements

- Original tutorial: [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial) by [Jeongeun Park](https://github.com/jeongeun980906).
- SO-ARM assets: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100).
- LeRobot: [huggingface/lerobot](https://github.com/huggingface/lerobot).
- MuJoCo parser inspiration: [yet-another-mujoco-tutorial](https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3).

## License

See [LICENSE](LICENSE). Third-party components (e.g. SO-ARM100, LeRobot) have their own licenses.
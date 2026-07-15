#!/usr/bin/env python3
"""Convert a *local* LeRobot dataset from codebase v2.1 to v3.0.

This is a small wrapper around Hugging Face LeRobot's upstream converter
(`convert_dataset` in `lerobot.scripts.convert_dataset_v21_to_v30`). Use it to
migrate **legacy on-disk v2.1** folders; Docker images in this repo ship
`lerobot==0.5.1`, which **records** v3.0 by default but can still run this
script for old datasets.

Prerequisites on disk (under ``--root``):

- ``meta/info.json`` with ``codebase_version`` set to ``v2.1``
- ``meta/episodes.jsonl``, ``meta/tasks.jsonl``, ``meta/episodes_stats.jsonl``
  (``meta/stats.json`` is optional for conversion; upstream recomputes global stats)
- ``data/`` (and ``videos/`` if your dataset used encoded video)

Some v2.1 episode parquet stores ``spawn.block_xyz`` as a length-4 NumPy **object**
vector of ``(3,)`` float arrays, which makes ``pyarrow.Schema.from_pandas`` fail
during conversion. This wrapper **rewrites those cells to nested Python lists**
(4×3 floats) before calling upstream unless ``--no-repair-parquet`` is set.

What the upstream converter does with a local ``--root``:

1. Writes the v3.0 tree under a sibling directory ``{dataset_name}_v30``.
2. Moves the original v2.1 tree to ``{dataset_name}_old``.
3. Moves the v3.0 tree into your original ``--root`` path (so the dataset path
   you passed is now v3.0).

Example:

    python -m venv .venv-convert && source .venv-convert/bin/activate
    pip install -U "lerobot>=0.5"
    python scripts/convert_lerobot_dataset_v21_to_v30_local.py \\
        --root ./demo_data \\
        --repo-id local/demo_data
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SPAWN_BLOCK_XYZ_COL = "spawn.block_xyz"


def _needs_spawn_block_xyz_repair(value: Any) -> bool:
    """True when stored as a 1-D numpy object array of four (3,) vectors (breaks PyArrow)."""
    if value is None:
        return False
    arr = np.asarray(value, dtype=object)
    if arr.ndim != 1 or arr.shape[0] != 4:
        return False
    if arr.dtype != object:
        return False
    try:
        stacked = np.stack([np.asarray(x, dtype=np.float32) for x in arr])
    except (TypeError, ValueError):
        return False
    return stacked.shape == (4, 3)


def _spawn_block_xyz_to_nested_list(value: Any) -> list[list[float]]:
    """4×3 nested Python list (float), matching v2.1 info.json feature shape."""
    if isinstance(value, list) and len(value) == 4:
        return [np.asarray(row, dtype=np.float32).tolist() for row in value]
    arr_obj = np.asarray(value, dtype=object)
    if arr_obj.ndim == 1 and arr_obj.shape[0] == 4 and arr_obj.dtype == object:
        return [np.asarray(x, dtype=np.float32).tolist() for x in arr_obj]
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (4, 3):
        return arr.tolist()
    if arr.size == 12:
        return arr.reshape(4, 3).tolist()
    raise ValueError(f"Unexpected {_SPAWN_BLOCK_XYZ_COL} value / shape: {value!r}")


def _repair_spawn_block_xyz_series(series: pd.Series) -> pd.Series:
    def _fix_cell(v: Any) -> Any:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return v
        if not _needs_spawn_block_xyz_repair(v):
            return v
        return _spawn_block_xyz_to_nested_list(v)

    return series.map(_fix_cell)


def repair_v21_data_parquet_for_v30_conversion(root: Path) -> int:
    """
    Rewrite broken `spawn.block_xyz` cells in v2.1 episode parquet so upstream
    `pa.Schema.from_pandas` succeeds during concat. Returns number of files modified.
    """
    data_dir = root / "data"
    if not data_dir.is_dir():
        return 0

    modified = 0
    for path in sorted(data_dir.rglob("*.parquet")):
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if _SPAWN_BLOCK_XYZ_COL not in df.columns:
            continue

        col = df[_SPAWN_BLOCK_XYZ_COL]
        sample: Any = None
        for v in col:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            sample = v
            break
        if sample is None or not _needs_spawn_block_xyz_repair(sample):
            continue

        df = df.copy()
        df[_SPAWN_BLOCK_XYZ_COL] = _repair_spawn_block_xyz_series(df[_SPAWN_BLOCK_XYZ_COL])
        df.to_parquet(path, index=False)
        modified += 1

    return modified


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dataset_root_hints(wrong_root: Path) -> str:
    """If user passed a repo or parent folder, suggest likely dataset paths."""
    lines: list[str] = []
    candidates: list[Path] = []
    for base in (wrong_root, wrong_root / "lerobot_mujoco_sim"):
        data_dir = base / "data"
        if not data_dir.is_dir():
            continue
        for p in sorted(data_dir.iterdir()):
            if p.is_dir() and (p / "meta" / "info.json").is_file():
                candidates.append(p.resolve())
    if candidates:
        lines.append(
            "Hint: `--root` must be the dataset directory that directly contains "
            "`meta/` and `data/` (not the git repo root or `/workspace`). Examples:"
        )
        for p in candidates[:5]:
            lines.append(f"  --root {p}")
        if len(candidates) > 5:
            lines.append(f"  … and {len(candidates) - 5} more under data/")
    else:
        lines.append(
            "Hint: pass the folder that contains `meta/info.json`. "
            "Docker often mounts the git repo at `/workspace`, so use e.g. "
            "`--root /workspace/data/<dataset>` (not `.../lerobot_mujoco_sim/data/...` "
            "unless that path exists in your container)."
        )
    return "\n".join(lines)


def _preflight_v21(root: Path) -> None:
    if not root.is_dir():
        msg = f"Not a directory: {root}"
        if "lerobot_mujoco_sim" in str(root):
            msg += (
                "\nHint: With `-v \"$PWD:/workspace\"` from inside `lerobot_mujoco_sim/`, "
                "the repo root is `/workspace`, not `/workspace/lerobot_mujoco_sim`. "
                "Use e.g. `--root /workspace/data/<dataset>`."
            )
        raise SystemExit(msg)

    required_meta = [
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "meta/episodes_stats.jsonl",
    ]
    missing = [p for p in required_meta if not (root / p).is_file()]
    if missing:
        hint = ""
        if not (root / "meta" / "info.json").is_file():
            hint = "\n" + _dataset_root_hints(root) + "\n"
        raise SystemExit(
            f"{root}: missing required files (wrong or incomplete `--root`?):\n  "
            + "\n  ".join(missing)
            + "\n"
            + hint
            + "\n"
            "Datasets recorded with this project should call `dataset.finalize()`; "
            "if `meta/episodes_stats.jsonl` is missing, record/finalize with a "
            "v2.1-capable LeRobot or add stats before converting. "
            "`--repo-id` is only an identifier for upstream (e.g. `local/my_run`), "
            "not the dataset path."
        )

    info = _load_json(root / "meta/info.json")
    ver = info.get("codebase_version")
    if ver != "v2.1":
        raise SystemExit(
            f"{root}/meta/info.json has codebase_version={ver!r}; "
            "this wrapper is only for v2.1 → v3.0."
        )

    if not (root / "data").is_dir():
        raise SystemExit(f"Expected a data/ directory under {root}")


def _get_convert_dataset():
    try:
        from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset
    except ImportError:
        try:
            from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
                convert_dataset,
            )
        except ImportError as exc:
            raise SystemExit(
                "Could not import LeRobot's v2.1→v3.0 converter.\n"
                "Install a recent LeRobot in this environment, e.g.:\n"
                "  pip install -U 'lerobot>=0.5'\n"
                f"Original error: {exc}"
            ) from exc
    return convert_dataset


def _init_lerobot_logging() -> None:
    """Match upstream CLI so INFO logs and inner tqdm bars are visible."""
    try:
        from lerobot.utils.utils import init_logging

        init_logging()
    except Exception:
        pass


def _run_convert_with_progress(
    convert_fn: Callable[..., None], *, kwargs: dict[str, Any]
) -> None:
    """Run blocking conversion while showing an outer elapsed-time tqdm bar."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print(
            "[convert] tqdm not installed; install tqdm or use pip's lerobot "
            "(it depends on tqdm). Proceeding without outer progress bar.",
            flush=True,
        )
        convert_fn(**kwargs)
        return

    stop = threading.Event()

    def _refresh_outer(pbar: Any) -> None:
        while not stop.wait(0.25):
            pbar.refresh()

    with tqdm(
        desc="v2.1→v3.0 (overall)",
        bar_format="{desc} | {elapsed} elapsed",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
    ) as outer:
        refresher = threading.Thread(target=_refresh_outer, args=(outer,), daemon=True)
        refresher.start()
        try:
            convert_fn(**kwargs)
        finally:
            stop.set()
            refresher.join(timeout=2.0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert a local LeRobot v2.1 dataset to v3.0 (wraps upstream LeRobot)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to the dataset root (directory containing meta/, data/, …).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Dummy or real HF repo id (e.g. local/my_run). Required by upstream; "
        "defaults to local/<folder_name>. Only used for Hub push if enabled.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="If set, push after conversion (needs HF auth and a real repo-id).",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Hub branch when pushing (upstream default: main).",
    )
    parser.add_argument(
        "--data-file-size-in-mb",
        type=int,
        default=None,
        help="Target shard size for parquet (upstream default ~100).",
    )
    parser.add_argument(
        "--video-file-size-in-mb",
        type=int,
        default=None,
        help="Target shard size for video (upstream default ~500).",
    )
    parser.add_argument(
        "--force-conversion",
        action="store_true",
        help="Skip upstream check for an existing Hub v3.0 snapshot.",
    )
    parser.add_argument(
        "--no-repair-parquet",
        action="store_true",
        help="Do not rewrite spawn.block_xyz in data/**/*.parquet (only if you know "
        "cells are already PyArrow-safe).",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Do not validate paths/version before calling upstream (not recommended).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the outer tqdm elapsed bar (inner upstream tqdm/log lines unchanged).",
    )
    parser.add_argument(
        "--no-logging-setup",
        action="store_true",
        help="Do not call lerobot init_logging() (quieter if you only want tqdm).",
    )

    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not args.skip_preflight:
        print(f"[convert] Preflight: checking {root} …", flush=True)
        _preflight_v21(root)
        print("[convert] Preflight OK.", flush=True)

    if not args.no_repair_parquet:
        n_fixed = repair_v21_data_parquet_for_v30_conversion(root)
        if n_fixed:
            print(
                f"[convert] Repaired {n_fixed} parquet file(s) "
                f"({_SPAWN_BLOCK_XYZ_COL} for PyArrow). "
                "Use --no-repair-parquet to skip this step.",
                flush=True,
            )

    repo_id = args.repo_id or f"local/{root.name}"

    convert_dataset = _get_convert_dataset()
    ckwargs = dict(
        repo_id=repo_id,
        branch=args.branch,
        data_file_size_in_mb=args.data_file_size_in_mb,
        video_file_size_in_mb=args.video_file_size_in_mb,
        root=str(root),
        push_to_hub=args.push_to_hub,
        force_conversion=args.force_conversion,
    )

    print(
        f"[convert] Starting LeRobot upstream conversion "
        f"(repo_id={repo_id!r}, push_to_hub={args.push_to_hub})…",
        flush=True,
    )
    print(
        "[convert] Expect tqdm bars for data/video steps below; "
        "the overall line shows elapsed time.",
        flush=True,
    )

    if not args.no_logging_setup:
        _init_lerobot_logging()

    if args.no_progress:
        convert_dataset(**ckwargs)
    else:
        _run_convert_with_progress(convert_dataset, kwargs=ckwargs)

    print(f"[convert] Done. v3.0 dataset is at: {root}", flush=True)
    print(
        f"[convert] v2.1 backup (if needed): {root.parent / f'{root.name}_old'}",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

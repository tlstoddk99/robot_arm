#!/usr/bin/env python
# Minimal replay of one episode without LeRobotDataset

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from io import BytesIO
from PIL import Image

# Make project code importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mujoco_env.y_env import SimpleEnv  # noqa: E402
from glob import glob


def _decode_image_column(col):
    """Convert a list of HF image entries (dicts) into a (T, H, W, 3) uint8 array."""
    frames = []
    for entry in col:
        if isinstance(entry, dict):
            if "bytes" in entry:
                img = Image.open(BytesIO(entry["bytes"]))
                arr = np.array(img, dtype=np.uint8)
            elif "array" in entry:
                arr = np.array(entry["array"], dtype=np.uint8)
            else:
                raise ValueError(f"Unexpected image entry format: {entry.keys()}")
        else:
            arr = np.array(entry, dtype=np.uint8)
        frames.append(arr)
    return np.stack(frames, axis=0)


def load_all_parquet_episodes(root: Path):
    """
    Load all parquet shards under <root>/data/**/episode_*.parquet.

    Returns:
        episodes: list of dicts, one per episode, each with keys:
            - "observation.image": (T, H, W, 3) uint8
            - "observation.wrist_image": (T, H, W, 3) uint8
            - "action": (T, A) float32
            - "obj_init": (T, 6) float32
            - "spawn.block_xyz": (T, 4, 3) float32
    """
    data_dir = root / "data"
    # Recursively find all episode parquet files under any chunk-* subdirectory.
    parquet_paths = sorted(
        Path(p).resolve()
        for p in glob(str(data_dir / "**" / "episode_*.parquet"), recursive=True)
    )
    if not parquet_paths:
        raise FileNotFoundError(
            f"No parquet files found under {data_dir}/chunk-*/episode_*.parquet"
        )

    print(f"[minimal_replay] found {len(parquet_paths)} parquet files:")
    for p in parquet_paths:
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = p
        print("  ", rel)

    episodes: list[dict[str, np.ndarray]] = []
    for path in parquet_paths:
        table = pq.read_table(path)
        cols = table.to_pydict()

        obs_image = _decode_image_column(cols["observation.image"])
        wrist_image = _decode_image_column(cols["observation.wrist_image"])
        action = np.stack(cols["action"])
        obj_init = np.stack(cols["obj_init"])
        spawn_block_xyz = np.stack(cols["spawn.block_xyz"])

        episodes.append({
            "source_path": path,
            "observation.image": obs_image,
            "observation.wrist_image": wrist_image,
            "action": action,
            "obj_init": obj_init,
            "spawn.block_xyz": spawn_block_xyz,
        })

    return episodes


def main():
    # Project root that contains /data with one or more datasets
    # (e.g. /workspace/data/demo_data_so101/data/chunk-000/episode_*.parquet).
    # We want to replay *all* parquet files nested within /workspace/data/.
    dataset_root = PROJECT_ROOT
    episodes = load_all_parquet_episodes(dataset_root)
    n_episodes = len(episodes)
    current_ep = 0
    data = episodes[current_ep]
    print(f"[minimal_replay] starting from episode 0 ({data['source_path']})")
    T = data["action"].shape[0]

    # Build environment (same scene as collect script). For visualization we
    # immediately overwrite object poses from the dataset, so we only need a
    # spawn workspace that allows reset to succeed without overly strict
    # spacing. Use the same reachable region as scripted collection.
    xml_path = str(PROJECT_ROOT / "asset" / "scene_so101_y.xml")
    env = SimpleEnv(
        xml_path,
        # We'll reconstruct absolute joint angles from the stored joint deltas
        # and replay them in joint_angle mode for robustness.
        action_type="joint_angle",
        # Target object to pick: blue block
        pick_body_name="body_obj_block_2",
        place_body_name="body_obj_bin",
        spawn_x_range=(0.21, 0.27),
        spawn_y_range=(0.04, 0.16),
        spawn_z_range=(0.815, 0.815),
        spawn_min_dist=0.04,
        spawn_xy_margin=0.0,
        spawn_fallback_min_dist=0.04,
    )

    step = 0
    env.reset()

    while env.env.is_viewer_alive():
        env.step_env()
        if env.env.loop_every(HZ=20):
            if step >= T:
                # Advance to next episode and reset environment
                current_ep = (current_ep + 1) % n_episodes
                data = episodes[current_ep]
                print(f"[minimal_replay] switching to episode {current_ep} ({data['source_path']})")
                T = data["action"].shape[0]
                step = 0
                env.reset()

            if step == 0:
                # Restore pick/place object poses from dataset
                obj0 = data["obj_init"][0]  # [6]
                p_pick = obj0[:3]
                p_place = obj0[3:]
                env.set_obj_pose(p_pick, p_place)

                # Restore full block spawn configuration
                spawn_xyz = data["spawn.block_xyz"][0]  # (4, 3)
                all_obj_names = env.env.get_body_names(prefix="body_obj_")
                obj_names = [n for n in all_obj_names if n != env.place_body_name]
                if spawn_xyz.shape[0] == len(obj_names):
                    for idx, body_name in enumerate(obj_names):
                        env.env.set_p_base_body(body_name=body_name, p=spawn_xyz[idx, :])
                        env.env.set_R_base_body(body_name=body_name, R=np.eye(3, 3))

            # Apply action: stored "action" is absolute joint angles + gripper
            # (same as 1_batch_collect_data and 1.collect_data). Use joint_angle
            # mode and pass the recorded frame directly — do not integrate.
            act = np.asarray(data["action"][step], dtype=np.float32)
            env.step(act)

            # Feed images into overlay buffers
            rgb_top = data["observation.image"][step]
            rgb_front = data["observation.wrist_image"][step]
            env.rgb_agent = rgb_top
            env.rgb_ego = rgb_front
            env.rgb_side = rgb_front.copy()

            env.render()
            step += 1

    env.env.close_viewer()


if __name__ == "__main__":
    main()
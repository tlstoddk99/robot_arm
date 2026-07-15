#!/usr/bin/env python
######################################################################################
# Collect Demonstration from Keyboard

# Collect demonstration data for the given environment.
# The task is to pick an object and place in a designated location. The environment recognizes the success if the object is in the designated location, the gripper is opened, and the end-effector is positioned above the object.

# Use WASD for x/y movement, RF for z movement, QE for yaw, LEFT/RIGHT for roll, and UP/DOWN for pitch.
# SPACEBAR will change your gripper's state, and Z key will reset your environment with discarding the current episode data.

# For overlayed images, 
# - Top Right: Agent View 
# - Bottom Right: Egocentric View
# - Top Left: Left Side View
# - Bottom Left: Top View

import sys
from pathlib import Path

# Get project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import numpy as np
import os
from PIL import Image
import shutil
from dataclasses import dataclass
import yaml
from mujoco_env.y_env import SimpleEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset

@dataclass
class CollectConfig:
    seed: int | None = None
    repo_name: str = 'so101_pnp'
    num_demo: int = 1
    root: str = str(PROJECT_ROOT / 'data' / 'demo_data_so101')
    task_name: str = 'Put blue block in the bin'
    xml_path: str = str(PROJECT_ROOT / 'asset' / 'scene_so101_y.xml')
    # Target object to pick: blue block (body_obj_block_2 in obj_blocks.xml)
    pick_body_name: str = 'body_obj_block_2'
    place_body_name: str = 'body_obj_bin'
    env_robot_profile: str = 'so101'
    offline_local_only: bool = True
    delete_existing_dataset: bool = False
    robot_type: str = 'so101'
    fps: int = 20
    image_size: int = 256
    image_writer_threads: int = 10
    image_writer_processes: int = 5
    cleanup_images: bool = True
    spawn_x_min: float = 0.25
    spawn_x_max: float = 0.52
    spawn_y_min: float = 0.02
    spawn_y_max: float = 0.22
    spawn_z_min: float = 0.815
    spawn_z_max: float = 0.815
    spawn_min_dist: float = 0.2
    spawn_xy_margin: float = 0.0
    spawn_fallback_min_dist: float = 0.1


def parse_args():
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "collect_manual.yaml"),
        help="Path to YAML config file.",
    )
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    default_cfg = CollectConfig()
    merged_defaults = default_cfg.__dict__.copy()
    if os.path.exists(bootstrap_args.config):
        with open(bootstrap_args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        unknown_keys = [k for k in yaml_cfg.keys() if k not in merged_defaults]
        if unknown_keys:
            raise ValueError(f"Unknown keys in config file {bootstrap_args.config}: {unknown_keys}")
        merged_defaults.update(yaml_cfg)
        print(f"[collect_data] Loaded config: {bootstrap_args.config}")
    else:
        print(f"[collect_data] Config not found: {bootstrap_args.config}. Using built-in defaults.")

    parser = argparse.ArgumentParser(
        description="Collect teleoperation demonstrations.",
        parents=[bootstrap_parser],
    )
    parser.add_argument("--seed", type=int, default=merged_defaults["seed"], help="Random seed. Use None for randomized object positions.")
    parser.add_argument("--repo-name", default=merged_defaults["repo_name"], help="LeRobot dataset repo id.")
    parser.add_argument("--num-demo", type=int, default=merged_defaults["num_demo"], help="Number of demonstrations to collect.")
    parser.add_argument("--root", default=merged_defaults["root"], help="Root directory to save demonstrations.")
    parser.add_argument("--task-name", default=merged_defaults["task_name"], help="Task name stored in dataset.")
    parser.add_argument("--xml-path", default=merged_defaults["xml_path"], help="MuJoCo scene XML path.")
    parser.add_argument("--pick-body-name", default=merged_defaults["pick_body_name"], help="Pick object body name in scene.")
    parser.add_argument("--place-body-name", default=merged_defaults["place_body_name"], help="Place/target body name in scene.")
    parser.add_argument("--env-robot-profile", default=merged_defaults["env_robot_profile"], choices=["omy", "so100", "so101"], help="Robot kinematic profile used by SimpleEnv.")
    parser.add_argument("--offline-local-only", action=argparse.BooleanOptionalAction, default=merged_defaults["offline_local_only"], help="Use local dataset metadata only and avoid any Hugging Face fallback calls.")
    parser.add_argument("--delete-existing-dataset", action=argparse.BooleanOptionalAction, default=merged_defaults["delete_existing_dataset"], help="Whether to delete existing dataset directory before collection.")
    parser.add_argument("--robot-type", default=merged_defaults["robot_type"], help="Robot type for LeRobotDataset.")
    parser.add_argument("--fps", type=int, default=merged_defaults["fps"], help="Dataset FPS.")
    parser.add_argument("--image-size", type=int, default=merged_defaults["image_size"], help="Square image size for saved frames.")
    parser.add_argument("--image-writer-threads", type=int, default=merged_defaults["image_writer_threads"], help="Image writer thread count.")
    parser.add_argument("--image-writer-processes", type=int, default=merged_defaults["image_writer_processes"], help="Image writer process count.")
    parser.add_argument("--cleanup-images", action=argparse.BooleanOptionalAction, default=merged_defaults["cleanup_images"], help="Whether to remove raw image directory after run.")
    parser.add_argument("--spawn-x-min", type=float, default=merged_defaults["spawn_x_min"], help="Minimum x for object spawn sampling.")
    parser.add_argument("--spawn-x-max", type=float, default=merged_defaults["spawn_x_max"], help="Maximum x for object spawn sampling.")
    parser.add_argument("--spawn-y-min", type=float, default=merged_defaults["spawn_y_min"], help="Minimum y for object spawn sampling.")
    parser.add_argument("--spawn-y-max", type=float, default=merged_defaults["spawn_y_max"], help="Maximum y for object spawn sampling.")
    parser.add_argument("--spawn-z-min", type=float, default=merged_defaults["spawn_z_min"], help="Minimum z for object spawn sampling.")
    parser.add_argument("--spawn-z-max", type=float, default=merged_defaults["spawn_z_max"], help="Maximum z for object spawn sampling.")
    parser.add_argument("--spawn-min-dist", type=float, default=merged_defaults["spawn_min_dist"], help="Minimum distance between spawned objects.")
    parser.add_argument("--spawn-xy-margin", type=float, default=merged_defaults["spawn_xy_margin"], help="XY margin inside spawn bounds.")
    parser.add_argument("--spawn-fallback-min-dist", type=float, default=merged_defaults["spawn_fallback_min_dist"], help="Fallback min distance if initial sampling fails.")

    args = parser.parse_args()
    config = CollectConfig(
        seed=args.seed,
        repo_name=args.repo_name,
        num_demo=args.num_demo,
        root=args.root,
        task_name=args.task_name,
        xml_path=args.xml_path,
        pick_body_name=args.pick_body_name,
        place_body_name=args.place_body_name,
        env_robot_profile=args.env_robot_profile,
        offline_local_only=args.offline_local_only,
        delete_existing_dataset=args.delete_existing_dataset,
        robot_type=args.robot_type,
        fps=args.fps,
        image_size=args.image_size,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        cleanup_images=args.cleanup_images,
        spawn_x_min=args.spawn_x_min,
        spawn_x_max=args.spawn_x_max,
        spawn_y_min=args.spawn_y_min,
        spawn_y_max=args.spawn_y_max,
        spawn_z_min=args.spawn_z_min,
        spawn_z_max=args.spawn_z_max,
        spawn_min_dist=args.spawn_min_dist,
        spawn_xy_margin=args.spawn_xy_margin,
        spawn_fallback_min_dist=args.spawn_fallback_min_dist,
    )

    if not os.path.isabs(config.xml_path):
        config.xml_path = str((PROJECT_ROOT / config.xml_path).resolve())

    return config


def resolve_robot_scene_defaults(config):
    scene_by_profile = {
        'omy': str(PROJECT_ROOT / 'asset' / 'scene_y.xml'),
        'so100': str(PROJECT_ROOT / 'asset' / 'scene_so100_y.xml'),
        'so101': str(PROJECT_ROOT / 'asset' / 'scene_so101_y.xml'),
    }
    root_by_profile = {
        'omy': str(PROJECT_ROOT / 'data' / 'demo_data'),
        'so100': str(PROJECT_ROOT / 'data' / 'demo_data_so100'),
        'so101': str(PROJECT_ROOT / 'data' / 'demo_data_so101'),
    }

    xml_norm = os.path.normpath(config.xml_path)
    default_xml_norms = {os.path.normpath(path) for path in scene_by_profile.values()}
    default_xml_suffixes = {
        os.path.normpath(os.path.join('asset', Path(path).name))
        for path in scene_by_profile.values()
    }
    is_default_xml = (
        xml_norm in default_xml_norms
        or any(xml_norm.endswith(suffix) for suffix in default_xml_suffixes)
    )
    if is_default_xml:
        config.xml_path = scene_by_profile[config.env_robot_profile]

    if not os.path.isabs(config.root):
        config.root = str((PROJECT_ROOT / config.root).resolve())

    default_root_norms = {os.path.normpath(path) for path in root_by_profile.values()}
    legacy_root_norms = {
        os.path.normpath(str(PROJECT_ROOT / 'demo_data')),
        os.path.normpath(str(PROJECT_ROOT / 'demo_data_so100')),
        os.path.normpath(str(PROJECT_ROOT / 'demo_data_so101')),
    }
    root_norm = os.path.normpath(config.root)
    if root_norm in default_root_norms or root_norm in legacy_root_norms:
        config.root = root_by_profile[config.env_robot_profile]

    return config


def validate_config(config):
    if config.num_demo <= 0:
        raise ValueError("--num-demo must be > 0")
    if config.fps <= 0:
        raise ValueError("--fps must be > 0")
    if config.image_size <= 0:
        raise ValueError("--image-size must be > 0")
    if config.image_writer_threads <= 0:
        raise ValueError("--image-writer-threads must be > 0")
    if config.image_writer_processes <= 0:
        raise ValueError("--image-writer-processes must be > 0")

    if config.spawn_x_min > config.spawn_x_max:
        raise ValueError("--spawn-x-min must be <= --spawn-x-max")
    if config.spawn_y_min > config.spawn_y_max:
        raise ValueError("--spawn-y-min must be <= --spawn-y-max")
    if config.spawn_z_min > config.spawn_z_max:
        raise ValueError("--spawn-z-min must be <= --spawn-z-max")

    if config.spawn_min_dist < 0:
        raise ValueError("--spawn-min-dist must be >= 0")
    if config.spawn_fallback_min_dist < 0:
        raise ValueError("--spawn-fallback-min-dist must be >= 0")
    if config.spawn_xy_margin < 0:
        raise ValueError("--spawn-xy-margin must be >= 0")

    x_span = config.spawn_x_max - config.spawn_x_min
    y_span = config.spawn_y_max - config.spawn_y_min
    if 2 * config.spawn_xy_margin >= x_span:
        raise ValueError("--spawn-xy-margin is too large for x range")
    if 2 * config.spawn_xy_margin >= y_span:
        raise ValueError("--spawn-xy-margin is too large for y range")


def build_env(config):
    action_type = 'delta_joint_angle' if config.env_robot_profile == 'so101' else 'eef_pose'
    
    return SimpleEnv(
        config.xml_path,
        action_type=action_type,
        robot_profile=config.env_robot_profile,
        seed=config.seed,
        state_type='joint_angle',
        pick_body_name=config.pick_body_name,
        place_body_name=config.place_body_name,
        spawn_x_range=(config.spawn_x_min, config.spawn_x_max),
        spawn_y_range=(config.spawn_y_min, config.spawn_y_max),
        spawn_z_range=(config.spawn_z_min, config.spawn_z_max),
        spawn_min_dist=config.spawn_min_dist,
        spawn_xy_margin=config.spawn_xy_margin,
        spawn_fallback_min_dist=config.spawn_fallback_min_dist,
    )


def print_controls(config):
    if config.env_robot_profile == 'so101':
        print("[controls] SO101 joint mode: A/D shoulder_pan (left/right), UP/DOWN wrist_flex (up/down), W/S shoulder_lift, R/F elbow, LEFT/RIGHT wrist_roll, SPACE gripper, Z reset")
    else:
        print("[controls] Cartesian mode: W/S +/-X, A/D +/-Y, R/F +/-Z, Q/E yaw, LEFT/RIGHT roll, UP/DOWN pitch, SPACE gripper, Z reset")


def create_or_load_dataset(config):
    action_dim = 7 if config.env_robot_profile == 'omy' else 6

    def has_minimal_local_meta(root_path: Path) -> bool:
        info_path = root_path / "meta" / "info.json"
        if not info_path.is_file():
            return False
        try:
            with info_path.open("r", encoding="utf-8") as f:
                ver = json.load(f).get("codebase_version", "")
        except (OSError, json.JSONDecodeError):
            return False
        if ver == "v3.0":
            if not (root_path / "meta" / "tasks.parquet").is_file():
                return False
            episodes_root = root_path / "meta" / "episodes"
            if not episodes_root.is_dir():
                return False
            return any(episodes_root.rglob("*.parquet"))
        legacy = [
            root_path / "meta" / "tasks.jsonl",
            root_path / "meta" / "episodes.jsonl",
        ]
        return all(p.is_file() for p in legacy)

    create_new = True
    root_path = Path(config.root)
    if root_path.exists():
        print(f"Directory {config.root} already exists.")
        if config.delete_existing_dataset:
            shutil.rmtree(root_path)
            print(f"Deleted existing dataset directory: {config.root}")
        else:
            if config.offline_local_only:
                print("[collect_data] Offline local-only mode is enabled; skipping existing dataset load and creating a fresh local dataset.")
            elif has_minimal_local_meta(root_path):
                try:
                    print("Load from previous dataset")
                    return LeRobotDataset(config.repo_name, root=config.root)
                except Exception as exc:
                    print(f"[collect_data] Failed to load existing dataset locally: {exc}")
                    print("[collect_data] Falling back to creating a fresh local dataset.")
            else:
                print("[collect_data] Existing directory is missing dataset metadata; creating a fresh local dataset.")

            suffix_root = f"{config.root}_fresh"
            suffix_idx = 0
            candidate = Path(suffix_root)
            while candidate.exists():
                suffix_idx += 1
                candidate = Path(f"{suffix_root}_{suffix_idx}")
            config.root = str(candidate)
            print(f"[collect_data] New dataset root: {config.root}")

    if create_new:
        return LeRobotDataset.create(
            repo_id=config.repo_name,
            root=config.root,
            robot_type=config.robot_type,
            fps=config.fps,
            features={
                # observation.image:   topview camera frames
                # observation.wrist_image: frontview camera frames
                "observation.image": {
                    "dtype": "image",
                    "shape": (config.image_size, config.image_size, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.wrist_image": {
                    "dtype": "image",
                    "shape": (config.image_size, config.image_size, 3),
                    "names": ["height", "width", "channel"],
                },
                # spawn.block_xyz: per-episode initial spawn positions of all
                # movable blocks (excluding the fixed bin/plate). In the SO101
                # tabletop scene this is always 4 blocks.
                "spawn.block_xyz": {
                    "dtype": "float32",
                    "shape": (4, 3),
                    "names": ["blocks", "xyz"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["state"],  # x, y, z, roll, pitch, yaw
                },
                "action": {
                    "dtype": "float32",
                    "shape": (action_dim,),
                    "names": ["action"],
                },
                "obj_init": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["obj_init"],  # initial object pose, not used in training
                },
            },
            image_writer_threads=config.image_writer_threads,
            image_writer_processes=config.image_writer_processes,
        )


def resize_images(agent_image, wrist_image, image_size):
    agent_image = Image.fromarray(agent_image).resize((image_size, image_size))
    wrist_image = Image.fromarray(wrist_image).resize((image_size, image_size))
    return np.array(agent_image), np.array(wrist_image)


def collect_demonstrations(env, dataset, config):
    action = np.zeros(7)
    episode_id = 0
    record_flag = False
    frames_in_episode = 0
    success_streak = 0
    success_streak_required = 6  # ~0.3s at 20Hz

    while env.env.is_viewer_alive() and episode_id < config.num_demo:
        env.step_env()
        if env.env.loop_every(HZ=20):
            success_now = bool(env.check_success())
            success_streak = (success_streak + 1) if success_now else 0
            done = success_streak >= success_streak_required
            
            # Debug: log success detection
            if success_now:
                print(f"[DEBUG] Success detected! streak={success_streak}/{success_streak_required}")
            
            if done:
                if frames_in_episode > 0:
                    dataset.save_episode()
                    episode_id += 1
                    print(f"[collect_data] Episode {episode_id-1} saved successfully (success detected)")
                else:
                    # Avoid crashing when success is triggered before any
                    # recorded frame exists (e.g., incidental early success).
                    dataset.clear_episode_buffer()
                    print("[collect_data] Success detected before recording frames; reset without saving.")
                print(f"[collect_data] RESET TRIGGERED BY SUCCESS (done=True, success_streak={success_streak})")
                env.reset(seed=config.seed)
                record_flag = False
                frames_in_episode = 0
                success_streak = 0

            action, reset = env.teleop_robot()
            if not record_flag and sum(action) != 0:
                record_flag = True
                print("Start recording")

            if reset:
                print(f"[collect_data] RESET TRIGGERED BY TELEOP (Z key pressed)")
                env.reset(seed=config.seed)
                dataset.clear_episode_buffer()
                record_flag = False
                frames_in_episode = 0
                success_streak = 0

            ee_pose = env.get_ee_pose()
            agent_image, wrist_image = env.grab_image()
            agent_image, wrist_image = resize_images(agent_image, wrist_image, config.image_size)
            joint_q = env.step(action)

            if record_flag:
                dataset.add_frame(
                    {
                        "observation.image": agent_image,
                        "observation.wrist_image": wrist_image,
                        "observation.state": ee_pose,
                        "action": joint_q,
                        "obj_init": env.obj_init_pose,
                        # Store the initial block spawn configuration so that
                        # replay can reconstruct the exact scene.
                        "spawn.block_xyz": env.spawn_obj_xyzs.astype(np.float32),
                        "task": config.task_name,
                    },
                )
                frames_in_episode += 1

            env.render(teleop=True)


def cleanup_dataset_images(dataset, cleanup_images):
    if not cleanup_images:
        return

    images_dir = dataset.root / 'images'
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)


def main():
    config = parse_args()
    config = resolve_robot_scene_defaults(config)
    validate_config(config)
    print(f"[collect_data] env_robot_profile={config.env_robot_profile}, xml_path={config.xml_path}")
    print_controls(config)
    env = build_env(config)
    dataset = create_or_load_dataset(config)

    try:
        collect_demonstrations(env, dataset, config)
    finally:
        # IMPORTANT: finalize the dataset so that all parquet/video writers
        # are closed and v3 metadata is flushed before any cleanup happens.
        try:
            dataset.finalize()
        except Exception:
            # If finalize fails, still attempt to close viewer and clean up images.
            pass
        env.env.close_viewer()
        cleanup_dataset_images(dataset, config.cleanup_images)


if __name__ == "__main__":
    main()
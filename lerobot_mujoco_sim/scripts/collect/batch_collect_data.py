#!/usr/bin/env python
######################################################################################
# Collect Demonstration from Keyboard

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import numpy as np
import os
import time
from PIL import Image
import shutil
from dataclasses import dataclass
import yaml
from mujoco_env.y_env import SimpleEnv
from mujoco_env.ik import solve_ik
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-untyped]
from so101.inverse_kinematics import get_inverse_kinematics
from so101.mujoco_utils import move_to_pose

from controllers.scripted_fsm_controller import setup_so101_controller, fsm_step, PHASES


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
    control_hz: int = 40
    image_size: int = 256
    image_writer_threads: int = 10
    image_writer_processes: int = 5
    cleanup_images: bool = True
    spawn_x_min: float = 0.21
    spawn_x_max: float = 0.27
    spawn_y_min: float = 0.04
    spawn_y_max: float = 0.16
    spawn_z_min: float = 0.815
    spawn_z_max: float = 0.815
    spawn_min_dist: float = 0.01
    spawn_xy_margin: float = 0.0
    spawn_fallback_min_dist: float = 0.1
    headless: bool = False
    ee_site_name: str = 'gripperframe'  # End-effector site name in XML
    arm_joint_names: list = None  # Will be set based on robot profile


def parse_args():
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "collect_batch.yaml"),
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
    parser.add_argument("--seed", type=int, default=merged_defaults["seed"])
    parser.add_argument("--repo-name", default=merged_defaults["repo_name"])
    parser.add_argument("--num-demo", type=int, default=merged_defaults["num_demo"])
    parser.add_argument("--root", default=merged_defaults["root"])
    parser.add_argument("--task-name", default=merged_defaults["task_name"])
    parser.add_argument("--xml-path", default=merged_defaults["xml_path"])
    parser.add_argument("--pick-body-name", default=merged_defaults["pick_body_name"])
    parser.add_argument("--place-body-name", default=merged_defaults["place_body_name"])
    parser.add_argument("--env-robot-profile", default=merged_defaults["env_robot_profile"], choices=["omy", "so100", "so101"])
    parser.add_argument("--offline-local-only", action=argparse.BooleanOptionalAction, default=merged_defaults["offline_local_only"])
    parser.add_argument("--delete-existing-dataset", action=argparse.BooleanOptionalAction, default=merged_defaults["delete_existing_dataset"])
    parser.add_argument("--robot-type", default=merged_defaults["robot_type"])
    parser.add_argument("--fps", type=int, default=merged_defaults["fps"])
    parser.add_argument("--control-hz", type=int, default=merged_defaults["control_hz"], help="Control loop rate (higher = faster arm motion)")
    parser.add_argument("--image-size", type=int, default=merged_defaults["image_size"])
    parser.add_argument("--image-writer-threads", type=int, default=merged_defaults["image_writer_threads"])
    parser.add_argument("--image-writer-processes", type=int, default=merged_defaults["image_writer_processes"])
    parser.add_argument("--cleanup-images", action=argparse.BooleanOptionalAction, default=merged_defaults["cleanup_images"])
    parser.add_argument("--spawn-x-min", type=float, default=merged_defaults["spawn_x_min"])
    parser.add_argument("--spawn-x-max", type=float, default=merged_defaults["spawn_x_max"])
    parser.add_argument("--spawn-y-min", type=float, default=merged_defaults["spawn_y_min"])
    parser.add_argument("--spawn-y-max", type=float, default=merged_defaults["spawn_y_max"])
    parser.add_argument("--spawn-z-min", type=float, default=merged_defaults["spawn_z_min"])
    parser.add_argument("--spawn-z-max", type=float, default=merged_defaults["spawn_z_max"])
    parser.add_argument("--spawn-min-dist", type=float, default=merged_defaults["spawn_min_dist"])
    parser.add_argument("--spawn-xy-margin", type=float, default=merged_defaults["spawn_xy_margin"])
    parser.add_argument("--spawn-fallback-min-dist", type=float, default=merged_defaults["spawn_fallback_min_dist"])
    parser.add_argument("--ee-site-name", default=merged_defaults["ee_site_name"])
    parser.add_argument("--arm-joint-names", nargs='+', default=None)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=merged_defaults.get("headless", False),
        help="Run without an on-screen viewer (no window rendering).",
    )

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
        control_hz=args.control_hz,
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
        headless=args.headless,
        ee_site_name=args.ee_site_name,
        arm_joint_names=args.arm_joint_names,
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
    if config.control_hz <= 0:
        raise ValueError("--control-hz must be > 0")
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


def enforce_reachable_spawn_workspace(config):
    if config.env_robot_profile != 'so101':
        return

    reachable = {
        'x_min': 0.21,
        'x_max': 0.27,
        'y_min': 0.04,
        'y_max': 0.16,
    }

    orig = (
        config.spawn_x_min,
        config.spawn_x_max,
        config.spawn_y_min,
        config.spawn_y_max,
    )

    config.spawn_x_min = max(config.spawn_x_min, reachable['x_min'])
    config.spawn_x_max = min(config.spawn_x_max, reachable['x_max'])
    config.spawn_y_min = max(config.spawn_y_min, reachable['y_min'])
    config.spawn_y_max = min(config.spawn_y_max, reachable['y_max'])

    if config.spawn_x_min > config.spawn_x_max:
        mid_x = 0.5 * (reachable['x_min'] + reachable['x_max'])
        config.spawn_x_min = mid_x
        config.spawn_x_max = mid_x
    if config.spawn_y_min > config.spawn_y_max:
        mid_y = 0.5 * (reachable['y_min'] + reachable['y_max'])
        config.spawn_y_min = mid_y
        config.spawn_y_max = mid_y

    now = (
        config.spawn_x_min,
        config.spawn_x_max,
        config.spawn_y_min,
        config.spawn_y_max,
    )
    if now != orig:
        print(
            "[collect_data] Clamped SO101 spawn workspace for reachability: "
            f"x[{orig[0]:.3f},{orig[1]:.3f}] -> [{now[0]:.3f},{now[1]:.3f}], "
            f"y[{orig[2]:.3f},{orig[3]:.3f}] -> [{now[2]:.3f},{now[3]:.3f}]"
        )


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
                "spawn.block_xyz": {
                    "dtype": "float32",
                    "shape": (4, 3),
                    "names": ["blocks", "xyz"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (action_dim,),
                    "names": ["action"],
                },
                "obj_init": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["obj_init"],
                },
            },
            image_writer_threads=config.image_writer_threads,
            image_writer_processes=config.image_writer_processes,
        )


def resize_images(agent_image, wrist_image, image_size):
    agent_image = Image.fromarray(agent_image).resize((image_size, image_size))
    wrist_image = Image.fromarray(wrist_image).resize((image_size, image_size))
    return np.array(agent_image), np.array(wrist_image)


def get_green_cube_location(env):
    p_green_cube, _ = env.get_obj_pose()
    return np.asarray(p_green_cube, dtype=np.float32)


# FIX: get_ee_pose() returns an incorrect position for the SO101 (~140mm error
# in z). Use the 'gripperframe' MuJoCo site directly for FSM control logic.
# get_ee_pose() is still used for dataset recording (observation.state) since
# that is what the policy will receive at inference time.
def _get_ee_xyz(env):
    site_id = env.env.model.site('gripperframe').id
    return env.env.data.site_xpos[site_id].copy().astype(np.float32)


def _joint_target_to_pose_dict(env, q_target_rad, gripper_fraction):
    q_deg = np.asarray(q_target_rad, dtype=np.float32) * (180.0 / np.pi)
    gripper_cmd_rad = float(env._make_gripper_ctrl(gripper_fraction)[0])
    gripper_util = gripper_cmd_rad * (100.0 / np.pi)
    return {
        'shoulder_pan': float(q_deg[0]),
        'shoulder_lift': float(q_deg[1]),
        'elbow_flex': float(q_deg[2]),
        'wrist_flex': float(q_deg[3]),
        'wrist_roll': float(q_deg[4]),
        'gripper': float(gripper_util),
    }


class _ViewerSyncAdapter:
    def __init__(self, viewer, render_callback=None, min_interval_s=1.0 / 30.0):
        self.viewer = viewer
        self.render_callback = render_callback
        self.min_interval_s = float(min_interval_s)
        self._last_render_time = 0.0

    def sync(self):
        if hasattr(self.viewer, 'sync'):
            self.viewer.sync()
        elif self.render_callback is not None:
            if hasattr(self.viewer, 'is_alive') and not self.viewer.is_alive:
                return
            now = time.monotonic()
            if now - self._last_render_time < self.min_interval_s:
                return
            try:
                self.render_callback()
                self._last_render_time = now
            except Exception:
                return


# REMOVED: Old IK functions (_ik_joint_targets_rad, _fallback_joint_targets_rad)
# These are no longer needed as mink handles IK internally


def _print_episode_stats(num_success: int, num_failed: int):
    """Print current success/failure counts and success percentage."""
    total = num_success + num_failed
    pct = (100.0 * num_success / total) if total else 0.0
    print(
        f"[collect_data] Episodes: {num_success} successful, {num_failed} unsuccessful, "
        f"{total} total — {pct:.1f}% success"
    )


def collect_demonstrations(env, dataset, config):
    episode_id = 0
    record_flag = True
    frames_in_episode = 0
    success_streak = 0
    success_streak_required = 6
    num_success = 0
    num_failed = 0

    # Set default joint names for SO101 if not provided
    if config.arm_joint_names is None:
        # SO-101 arm joint names (matches y_env.py robot_profile='so101')
        arm_joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    else:
        arm_joint_names = config.arm_joint_names
    
    fsm = setup_so101_controller(
        env=env,
        arm_joint_names=arm_joint_names,
        ee_site_name=config.ee_site_name
    )
    
    # fsm already includes traj_start/traj_target from setup function

    while env.env.is_viewer_alive() and episode_id < config.num_demo:
        env.step_env()
        if env.env.loop_every(HZ=config.control_hz):
            success_now = bool(env.check_success())
            success_streak = (success_streak + 1) if success_now else 0
            done = success_streak >= success_streak_required

            if fsm["tick"] == 0 or success_now:
                cube_pos, bin_pos = env.get_obj_pose()
                print(f"[DEBUG] fsm_phase={fsm['phase']} success_now={success_now} streak={success_streak} "
                      f"cube_xyz={np.round(cube_pos, 3)} bin_xyz={np.round(bin_pos, 3)}")

            if done:
                if frames_in_episode > 0:
                    dataset.save_episode()
                    episode_id += 1
                    num_success += 1
                    print(f"[collect_data] Episode {episode_id-1} saved successfully (success detected)")
                    _print_episode_stats(num_success, num_failed)
                else:
                    dataset.clear_episode_buffer()
                    print("[collect_data] Success detected before recording frames; reset without saving.")
                print(f"[collect_data] RESET TRIGGERED BY SUCCESS (done=True, success_streak={success_streak})")
                env.reset(seed=config.seed)
                record_flag = True
                frames_in_episode = 0
                success_streak = 0

                fsm = setup_so101_controller(
                    env=env,
                    arm_joint_names=arm_joint_names,
                    ee_site_name=config.ee_site_name
                )

            prev_phase = fsm["phase"]

            action = fsm_step(
                env=env,
                fsm=fsm,
                get_ee_xyz_fn=_get_ee_xyz,
                cube_pos_fn=get_green_cube_location,
            )

            if fsm["phase"] != prev_phase:
                print(f"[PHASE CHANGE] {prev_phase} -> {fsm['phase']} at frame {frames_in_episode}")

            # If we spend too long in the final "return_home" phase, treat the
            # episode as failed and reset.
            if (fsm["phase"] == len(PHASES) - 1) and fsm["tick"] > 80:
                dataset.clear_episode_buffer()
                num_failed += 1
                print("[collect_data] Retract timeout — episode failed, resetting.")
                _print_episode_stats(num_success, num_failed)
                env.reset(seed=config.seed)
                frames_in_episode = 0
                success_streak = 0

                fsm = setup_so101_controller(
                    env=env,
                    arm_joint_names=arm_joint_names,
                    ee_site_name=config.ee_site_name
                )

            # Apply the scripted action.
            env.step(action)

            # Record action in the same format as 1.collect_data.py:
            # absolute joint angles for all arm joints plus the normalized
            # gripper command in the last entry.
            joint_q = env.get_joint_state().astype(np.float32)

            ee_pose = env.get_ee_pose()  # kept for dataset recording only
            agent_image, wrist_image = env.grab_image()
            agent_image, wrist_image = resize_images(agent_image, wrist_image, config.image_size)

            if record_flag:
                # LeRobot 0.5+ writes frame PNGs under
                #   images/{feature}/episode-{episode_index:06d}/frame-{frame_index:06d}.png
                # DatasetWriter creates the directory on the first frame of each episode.
                dataset.add_frame(
                    {
                        "observation.image": agent_image,
                        "observation.wrist_image": wrist_image,
                        "observation.state": ee_pose,
                        "action": joint_q,
                        "obj_init": env.obj_init_pose,
                        "spawn.block_xyz": env.spawn_obj_xyzs.astype(np.float32),
                        "task": config.task_name,
                    },
                )
                frames_in_episode += 1

            if not config.headless:
                env.render(teleop=False)

    # Final summary when collection ends (target reached or viewer closed)
    if (num_success + num_failed) > 0:
        print("[collect_data] ---------- Final episode summary ----------")
        _print_episode_stats(num_success, num_failed)


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
    enforce_reachable_spawn_workspace(config)
    print(f"[collect_data] env_robot_profile={config.env_robot_profile}, xml_path={config.xml_path}")
    print_controls(config)
    env = build_env(config)

    if not config.headless:
        env.env.viewer.cam.lookat[0] = 0.35
        env.env.viewer.cam.lookat[1] = 0.0
        env.env.viewer.cam.lookat[2] = 0.85
        env.env.viewer.cam.distance = 1.5
        env.env.viewer.cam.azimuth = 180
        env.env.viewer.cam.elevation = -30

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
        if not config.headless:
            env.env.close_viewer()
        cleanup_dataset_images(dataset, config.cleanup_images)


if __name__ == "__main__":
    main()
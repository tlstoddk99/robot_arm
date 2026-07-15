#!/usr/bin/env python
"""
Roll out a trained ACT policy in the MuJoCo SO-101 pick-and-place environment.

Students should be able to:
- drop a checkpoint folder on disk (e.g. checkpoints/act_y/)
- run this script without editing code
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
import yaml

# Get project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from mujoco_env.y_env import SimpleEnv
from huggingface_hub import hf_hub_download


@dataclass
class DeployConfig:
    checkpoint: str = str(PROJECT_ROOT / "checkpoints" / "act_y")
    dataset_root: str = str(PROJECT_ROOT / "data" / "demo_data_so101")
    dataset_repo_id: str = "so101_pnp"
    xml_path: str = str(PROJECT_ROOT / "asset" / "scene_so101_y.xml")
    pick_body_name: str = "body_obj_block_2"
    place_body_name: str = "body_obj_bin"
    task: str = "Put blue block in the bin"
    hz: int = 20
    seed: int = 0
    device: str = "auto"
    image_size: int = 256
    spawn_x_min: float = 0.21
    spawn_x_max: float = 0.22
    spawn_y_min: float = 0.04
    spawn_y_max: float = 0.16
    spawn_z_min: float = 0.815
    spawn_z_max: float = 0.815
    spawn_min_dist: float = 0.01
    spawn_xy_margin: float = 0.0
    spawn_fallback_min_dist: float = 0.1
    num_trials: int = 20
    trial_timeout_s: float = 60.0


def parse_args() -> argparse.Namespace:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "deploy_act.yaml"),
        help="Path to YAML config file.",
    )
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    default_cfg = DeployConfig()
    merged_defaults = default_cfg.__dict__.copy()
    if os.path.exists(bootstrap_args.config):
        with open(bootstrap_args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        unknown_keys = [k for k in yaml_cfg.keys() if k not in merged_defaults]
        if unknown_keys:
            raise ValueError(f"Unknown keys in config file {bootstrap_args.config}: {unknown_keys}")
        merged_defaults.update(yaml_cfg)
        print(f"[deploy_act] Loaded config: {bootstrap_args.config}")
    else:
        print(f"[deploy_act] Config not found: {bootstrap_args.config}. Using built-in defaults.")

    p = argparse.ArgumentParser(description="Deploy a trained ACT policy in MuJoCo.")
    p.add_argument("--config", default=bootstrap_args.config, help="Path to YAML config file.")
    p.add_argument(
        "--checkpoint",
        default=merged_defaults["checkpoint"],
        help="Path to ACT checkpoint directory (created by ACTPolicy.save_pretrained).",
    )
    p.add_argument(
        "--dataset-root",
        default=merged_defaults["dataset_root"],
        help="Local dataset root used to load dataset stats/features for normalization.",
    )
    p.add_argument("--dataset-repo-id", default=merged_defaults["dataset_repo_id"], help="LeRobot dataset repo id used for metadata.")
    p.add_argument(
        "--xml-path",
        default=merged_defaults["xml_path"],
        help="MuJoCo scene XML path.",
    )
    p.add_argument("--pick-body-name", default=merged_defaults["pick_body_name"])
    p.add_argument("--place-body-name", default=merged_defaults["place_body_name"])
    p.add_argument("--task", default=merged_defaults["task"])
    p.add_argument("--hz", type=int, default=merged_defaults["hz"], help="Control loop rate in Hz.")
    p.add_argument("--seed", type=int, default=merged_defaults["seed"])
    p.add_argument("--image-size", type=int, default=merged_defaults["image_size"])
    p.add_argument("--spawn-x-min", type=float, default=merged_defaults["spawn_x_min"])
    p.add_argument("--spawn-x-max", type=float, default=merged_defaults["spawn_x_max"])
    p.add_argument("--spawn-y-min", type=float, default=merged_defaults["spawn_y_min"])
    p.add_argument("--spawn-y-max", type=float, default=merged_defaults["spawn_y_max"])
    p.add_argument("--spawn-z-min", type=float, default=merged_defaults["spawn_z_min"])
    p.add_argument("--spawn-z-max", type=float, default=merged_defaults["spawn_z_max"])
    p.add_argument("--spawn-min-dist", type=float, default=merged_defaults["spawn_min_dist"])
    p.add_argument("--spawn-xy-margin", type=float, default=merged_defaults["spawn_xy_margin"])
    p.add_argument("--spawn-fallback-min-dist", type=float, default=merged_defaults["spawn_fallback_min_dist"])
    p.add_argument(
        "--num-trials",
        type=int,
        default=merged_defaults["num_trials"],
        help="Number of evaluation trials to run (0 = run indefinitely).",
    )
    p.add_argument(
        "--trial-timeout-s",
        type=float,
        default=merged_defaults["trial_timeout_s"],
        help="Maximum seconds per trial before it is counted as a timeout/failure.",
    )
    p.add_argument(
        "--device",
        default=merged_defaults["device"],
        choices=["auto", "cpu", "cuda"],
        help="Device to run policy on.",
    )
    return p.parse_args()


def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()

    deploy_meta_path = checkpoint_path / "deploy_metadata.json"
    dataset_stats = None
    features_raw = None
    if deploy_meta_path.exists():
        deploy_meta = json.loads(deploy_meta_path.read_text(encoding="utf-8"))
        dataset_stats = deploy_meta.get("stats")
        features_raw = deploy_meta.get("features")
        if deploy_meta.get("dataset_repo_id"):
            args.dataset_repo_id = deploy_meta["dataset_repo_id"]

    if dataset_stats is None or features_raw is None:
        dataset_root = Path(args.dataset_root).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = (PROJECT_ROOT / dataset_root).resolve()

        # Prefer local dataset metadata (avoid any HF lookups).
        # This repo's `data/demo_data_*` directories contain:
        #   meta/info.json  (includes `features`)
        #   meta/stats.json (includes normalization stats)
        info_path = dataset_root / "meta" / "info.json"
        stats_path = dataset_root / "meta" / "stats.json"
        if info_path.exists() and stats_path.exists():
            deploy_info = json.loads(info_path.read_text(encoding="utf-8"))
            features_raw = deploy_info.get("features")
            raw_stats = json.loads(stats_path.read_text(encoding="utf-8"))

            # LeRobot's Normalize expects per-feature stats where `mean/std/min/max`
            # are `np.ndarray` (or `torch.Tensor`), not Python lists.
            dataset_stats = {}
            for feature_name, feature_stats in raw_stats.items():
                dataset_stats[feature_name] = {}
                for stat_name, stat_value in feature_stats.items():
                    if isinstance(stat_value, list):
                        dataset_stats[feature_name][stat_name] = np.asarray(stat_value, dtype=np.float32)
                    else:
                        dataset_stats[feature_name][stat_name] = stat_value
        else:
            # Download only the dataset metadata pieces we need for ACT deployment.
            # Some LeRobot dataset repos (including your HF dataset) include:
            #   meta/info.json, meta/stats.json
            # but not necessarily the legacy file(s) that `LeRobotDatasetMetadata`
            # tries to read (e.g. `meta/tasks.jsonl`).
            token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")
            info_fpath = hf_hub_download(
                repo_id=args.dataset_repo_id,
                filename="meta/info.json",
                repo_type="dataset",
                token=token,
            )
            stats_fpath = hf_hub_download(
                repo_id=args.dataset_repo_id,
                filename="meta/stats.json",
                repo_type="dataset",
                token=token,
            )

            deploy_info = json.loads(Path(info_fpath).read_text(encoding="utf-8"))
            features_raw = deploy_info.get("features")
            raw_stats = json.loads(Path(stats_fpath).read_text(encoding="utf-8"))

            dataset_stats = {}
            for feature_name, feature_stats in raw_stats.items():
                dataset_stats[feature_name] = {}
                for stat_name, stat_value in feature_stats.items():
                    if isinstance(stat_value, list):
                        dataset_stats[feature_name][stat_name] = np.asarray(stat_value, dtype=np.float32)
                    else:
                        dataset_stats[feature_name][stat_name] = stat_value

    # Rebuild ACTConfig from the checkpoint's config.json, but only pass the
    # ACTConfig-relevant fields. This avoids schema mismatches where the
    # checkpoint config contains extra Hugging Face fields (e.g. repo_id).
    ckpt_cfg_path = checkpoint_path / "config.json"
    if not ckpt_cfg_path.exists():
        raise FileNotFoundError(f"Checkpoint config not found: {ckpt_cfg_path}")

    ckpt_cfg = json.loads(ckpt_cfg_path.read_text(encoding="utf-8"))

    input_features = {
        name: PolicyFeature(
            type=FeatureType[feat_cfg["type"]],
            shape=tuple(feat_cfg["shape"]),
        )
        for name, feat_cfg in ckpt_cfg["input_features"].items()
    }
    output_features = {
        name: PolicyFeature(
            type=FeatureType[feat_cfg["type"]],
            shape=tuple(feat_cfg["shape"]),
        )
        for name, feat_cfg in ckpt_cfg["output_features"].items()
    }

    normalization_mapping = {
        feat_type: NormalizationMode[mode]
        for feat_type, mode in ckpt_cfg.get("normalization_mapping", {}).items()
    }

    cfg = ACTConfig(
        n_obs_steps=int(ckpt_cfg.get("n_obs_steps", 1)),
        input_features=input_features,
        output_features=output_features,
        chunk_size=int(ckpt_cfg["chunk_size"]),
        n_action_steps=int(ckpt_cfg["n_action_steps"]),
        temporal_ensemble_coeff=ckpt_cfg.get("temporal_ensemble_coeff", None),
        normalization_mapping=normalization_mapping,
        vision_backbone=ckpt_cfg.get("vision_backbone", "resnet18"),
        pretrained_backbone_weights=ckpt_cfg.get("pretrained_backbone_weights", None),
        replace_final_stride_with_dilation=bool(ckpt_cfg.get("replace_final_stride_with_dilation", False)),
        pre_norm=bool(ckpt_cfg.get("pre_norm", False)),
        dim_model=int(ckpt_cfg.get("dim_model", 512)),
        n_heads=int(ckpt_cfg.get("n_heads", 8)),
        dim_feedforward=int(ckpt_cfg.get("dim_feedforward", 3200)),
        feedforward_activation=ckpt_cfg.get("feedforward_activation", "relu"),
        n_encoder_layers=int(ckpt_cfg.get("n_encoder_layers", 4)),
        n_decoder_layers=int(ckpt_cfg.get("n_decoder_layers", 1)),
        use_vae=bool(ckpt_cfg.get("use_vae", True)),
        latent_dim=int(ckpt_cfg.get("latent_dim", 32)),
        n_vae_encoder_layers=int(ckpt_cfg.get("n_vae_encoder_layers", 4)),
        dropout=float(ckpt_cfg.get("dropout", 0.1)),
        kl_weight=float(ckpt_cfg.get("kl_weight", 10.0)),
        optimizer_lr=float(ckpt_cfg.get("optimizer_lr", 1e-05)),
        optimizer_weight_decay=float(ckpt_cfg.get("optimizer_weight_decay", 0.0001)),
        optimizer_lr_backbone=float(ckpt_cfg.get("optimizer_lr_backbone", 1e-05)),
        # Override the checkpoint device with the runtime device selection.
        device=str(device),
        use_amp=bool(ckpt_cfg.get("use_amp", False)),
    )

    policy = ACTPolicy.from_pretrained(str(checkpoint_path), config=cfg, dataset_stats=dataset_stats)
    policy.to(device)
    policy.eval()

    env = SimpleEnv(
        args.xml_path,
        action_type="joint_angle",
        pick_body_name=args.pick_body_name,
        place_body_name=args.place_body_name,
        spawn_x_range=(args.spawn_x_min, args.spawn_x_max),
        spawn_y_range=(args.spawn_y_min, args.spawn_y_max),
        spawn_z_range=(args.spawn_z_min, args.spawn_z_max),
        spawn_min_dist=args.spawn_min_dist,
        spawn_xy_margin=args.spawn_xy_margin,
        spawn_fallback_min_dist=args.spawn_fallback_min_dist,
    )

    img_transform = torchvision.transforms.ToTensor()

    import time

    num_trials = args.num_trials          # 0 → run indefinitely
    timeout_s  = args.trial_timeout_s

    trial_idx      = 0
    num_success    = 0
    num_timeout    = 0

    step           = 0
    trial_start    = time.monotonic()

    env.reset(seed=args.seed + trial_idx)
    policy.reset()

    print(
        f"[deploy_act] Starting evaluation: {num_trials} trial(s), "
        f"{timeout_s:.0f}s timeout each. "
        f"(num_trials=0 → run indefinitely)"
    )

    while env.env.is_viewer_alive():
        if num_trials > 0 and trial_idx >= num_trials:
            break

        env.step_env()
        if not env.env.loop_every(HZ=args.hz):
            continue

        elapsed = time.monotonic() - trial_start

        # ── success ──────────────────────────────────────────────────────────
        if bool(env.check_success()):
            num_success += 1
            print(
                f"[deploy_act] Trial {trial_idx + 1}/{num_trials or '∞'} — "
                f"SUCCESS in {elapsed:.1f}s"
            )
            trial_idx  += 1
            step        = 0
            policy.reset()
            env.reset(seed=args.seed + trial_idx)
            trial_start = time.monotonic()
            continue

        # ── timeout ──────────────────────────────────────────────────────────
        if elapsed >= timeout_s:
            num_timeout += 1
            print(
                f"[deploy_act] Trial {trial_idx + 1}/{num_trials or '∞'} — "
                f"TIMEOUT ({timeout_s:.0f}s)"
            )
            trial_idx  += 1
            step        = 0
            policy.reset()
            env.reset(seed=args.seed + trial_idx)
            trial_start = time.monotonic()
            continue

        # ── inference step ───────────────────────────────────────────────────
        state = env.get_ee_pose()
        image, wrist_image = env.grab_image()

        image_t = img_transform(Image.fromarray(image).resize((args.image_size, args.image_size))).unsqueeze(0).to(device)
        wrist_t = img_transform(Image.fromarray(wrist_image).resize((args.image_size, args.image_size))).unsqueeze(0).to(device)

        batch = {
            "observation.state": torch.tensor([state], dtype=torch.float32, device=device),
            "observation.image": image_t,
            "observation.wrist_image": wrist_t,
            "task": [args.task],
            "timestamp": torch.tensor([step / float(args.hz)], dtype=torch.float32, device=device),
        }

        action = policy.select_action(batch)[0].detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32)
        _ = env.step(action)
        env.render()
        step += 1

    # ── final summary ─────────────────────────────────────────────────────────
    completed = trial_idx
    if completed > 0:
        num_fail = completed - num_success - num_timeout
        pct = 100.0 * num_success / completed
        print(
            f"\n[deploy_act] ══ Evaluation complete ══\n"
            f"  Trials completed : {completed}\n"
            f"  Successes        : {num_success}\n"
            f"  Timeouts         : {num_timeout}\n"
            f"  Other failures   : {num_fail}\n"
            f"  Success rate     : {num_success}/{completed} = {pct:.1f}%"
        )


if __name__ == "__main__":
    main()

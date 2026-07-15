#!/usr/bin/env python3
"""
Upload a folder to the Hugging Face Hub as a dataset repo.

Auth:
- Run `huggingface-cli login` inside the environment, or set `HF_TOKEN`.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, login, upload_folder
from huggingface_hub.errors import RepositoryNotFoundError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a local folder to Hugging Face Hub (dataset repo).")
    p.add_argument(
        "--folder",
        required=True,
        help="Local folder to upload (e.g. ./data/demo_data_so101).",
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help='Target Hub dataset repo id like "username/repo_name".',
    )
    p.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Hub repo type (default: dataset).",
    )
    p.add_argument(
        "--message",
        default="Upload dataset folder",
        help="Commit message.",
    )
    p.add_argument(
        "--create-if-missing",
        action="store_true",
        help="Create the target repo if it does not exist.",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="If creating, make the repo private.",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (defaults to HF_TOKEN env var).",
    )
    return p.parse_args()


def ensure_repo_exists(repo_id: str, repo_type: str, *, private: bool) -> None:
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        return
    except RepositoryNotFoundError:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)


def main() -> None:
    args = parse_args()

    folder_path = Path(args.folder).expanduser().resolve()
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    # Optional: authenticate explicitly if token provided.
    if args.token:
        login(token=args.token)

    if args.create_if_missing:
        ensure_repo_exists(args.repo_id, args.repo_type, private=args.private)

    upload_folder(
        folder_path=str(folder_path),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message=args.message,
    )

    print(f"[upload_hf] Uploaded {folder_path} -> {args.repo_type}:{args.repo_id}")


if __name__ == "__main__":
    main()

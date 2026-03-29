from __future__ import annotations

import os

import torch

from src.utils.hf_wandb import pull_from_hub, push_to_hub


def save_artifact(payload: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    return path


def load_artifact(path: str, device: str = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def maybe_pull_from_hf(source: str, local_path: str, repo_id: str | None = None, filename: str | None = None) -> str:
    target_filename = filename or os.path.basename(local_path)

    if source == "hf":
        if not repo_id or not target_filename:
            raise ValueError("HF source requires repo_id and filename")
        local_dir = os.path.dirname(local_path) or "checkpoints"
        return pull_from_hub(repo_id=repo_id, filename=target_filename, local_dir=local_dir)

    if source == "local":
        if os.path.exists(local_path):
            return local_path
        if repo_id and target_filename:
            local_dir = os.path.dirname(local_path) or "checkpoints"
            print(
                f"[artifacts] local checkpoint missing at {local_path}; "
                f"downloading {target_filename} from {repo_id}"
            )
            return pull_from_hub(repo_id=repo_id, filename=target_filename, local_dir=local_dir)
        raise FileNotFoundError(
            f"Local checkpoint not found at {local_path} and no Hugging Face fallback is configured"
        )

    raise ValueError(f"Unsupported checkpoint source: {source}")


def maybe_push_to_hf(path: str, enabled: bool, repo_id: str | None = None, filename: str | None = None) -> None:
    if not enabled:
        return
    if not repo_id:
        raise ValueError("push_to_hf is enabled but repo_id is missing")
    push_to_hub(path=path, repo_id=repo_id, path_in_repo=filename)

import os
import json

import torch
import wandb
from huggingface_hub import HfApi, hf_hub_download


def _resolve_hf_token(token: str | None = None) -> str | None:
    if token:
        return token
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HF_TOKEN_READ")
        or os.environ.get("HF_TOKEN_PUSH")
    )


def init_wandb(
    project: str,
    config: dict,
    name: str | None = None,
    dir: str | None = None,
) -> wandb.sdk.wandb_run.Run:
    run = wandb.init(project=project, config=config, name=name, dir=dir)
    url = getattr(run, "url", None)
    if url:
        print(f"[wandb] run url: {url}")
    return run


def log_wandb(metrics: dict, step: int | None = None) -> None:
    wandb.log(metrics, step=step)


def finish_wandb() -> None:
    wandb.finish()


def save_wandb_run_info(run: wandb.sdk.wandb_run.Run, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "run_id": getattr(run, "id", None),
        "run_name": getattr(run, "name", None),
        "run_url": getattr(run, "url", None),
        "dir": getattr(run, "dir", None),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def push_to_hub(
    path: str,
    repo_id: str,
    path_in_repo: str | None = None,
    token: str | None = None,
) -> str:
    token = _resolve_hf_token(token)
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
    return api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path_in_repo or os.path.basename(path),
        repo_id=repo_id,
        token=token,
    )


def pull_from_hub(
    repo_id: str,
    filename: str,
    local_dir: str = "checkpoints",
    token: str | None = None,
) -> str:
    token = _resolve_hf_token(token)
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        token=token,
    )


def save_and_push(
    model: torch.nn.Module,
    repo_id: str,
    filename: str = "model.pt",
    local_dir: str = "checkpoints",
    token: str | None = None,
) -> str:
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    torch.save(model.state_dict(), local_path)
    return push_to_hub(local_path, repo_id, filename, token)


def load_from_hub(
    model: torch.nn.Module,
    repo_id: str,
    filename: str = "model.pt",
    local_dir: str = "checkpoints",
    device: str = "cpu",
    token: str | None = None,
) -> torch.nn.Module:
    path = pull_from_hub(repo_id, filename, local_dir, token)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model

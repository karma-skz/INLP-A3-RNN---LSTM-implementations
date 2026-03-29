from __future__ import annotations

import json
import os

import torch

from src.common.artifacts import load_artifact, maybe_pull_from_hf, maybe_push_to_hf, save_artifact
from src.common.config import load_config
from src.common.io_utils import ensure_dir, pick_device, set_seed
from src.common.layers import ManualBiLSTMLM
from src.task2.common import build_mlm_dataloaders, run_mlm_epoch
from src.utils.hf_wandb import finish_wandb, init_wandb, log_wandb, save_wandb_run_info


def _build_model(model_config: dict, vocab_size: int):
    return ManualBiLSTMLM(
        vocab_size=vocab_size,
        emb_dim=model_config["emb_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config.get("num_layers", 1),
        dropout=model_config.get("dropout", 0.0),
        tie_embeddings=model_config.get("tie_embeddings", False),
    )


def _load_model_from_artifact(config: dict, device: str):
    model_path = maybe_pull_from_hf(
        source=config["checkpoint"]["source"],
        local_path=config["checkpoint"]["local_path"],
        repo_id=config["checkpoint"].get("repo_id"),
        filename=config["checkpoint"].get("hf_filename"),
    )
    artifact = load_artifact(model_path, device=device)
    model = _build_model(artifact["config"], artifact["vocab_size"]).to(device)
    model.load_state_dict(artifact["model_state"])
    return model, artifact["vocab"]


def main(config_path: str, mode: str = "evaluate"):
    config = load_config(config_path)
    set_seed(config["seed"])
    device = pick_device(config["train"].get("use_cuda", True))
    wandb_cfg = config.get("wandb", {})
    wandb_run = None
    if wandb_cfg.get("enabled", False):
        wandb_run = init_wandb(
            project=wandb_cfg.get("project", "inlp-a3"),
            config=config,
            name=wandb_cfg.get("run_name", "task2_bilstm"),
            dir=wandb_cfg.get("dir"),
        )

    ensure_dir(config["paths"]["results_dir"])
    ensure_dir(os.path.dirname(config["checkpoint"]["local_path"]))
    if wandb_run is not None:
        save_wandb_run_info(wandb_run, os.path.join(config["paths"]["results_dir"], "task2_bilstm_wandb.json"))

    if mode in {"train", "both"}:
        train_cfg = dict(config)
        train_cfg["train"] = dict(config["train"])
        while True:
            try:
                train_loader, val_loader, vocab = build_mlm_dataloaders(train_cfg)
                model = _build_model(config["model"], len(vocab["itos"])).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config["train"]["lr"],
                    weight_decay=config["train"].get("weight_decay", 0.0),
                )
                best_val = float("inf")
                grad_clip = config["train"].get("grad_clip")

                for epoch in range(1, config["train"]["epochs"] + 1):
                    train_loss, train_ppl = run_mlm_epoch(
                        model,
                        train_loader,
                        optimizer,
                        device,
                        grad_clip=grad_clip,
                    )
                    val_loss, val_ppl = run_mlm_epoch(model, val_loader, None, device)
                    print(
                        f"[task2_bilstm] epoch={epoch} train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} "
                        f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}"
                    )
                    if wandb_run is not None:
                        log_wandb(
                            {
                                "train_loss": train_loss,
                                "train_perplexity": train_ppl,
                                "val_loss": val_loss,
                                "val_perplexity": val_ppl,
                            },
                            step=epoch,
                        )
                    if val_loss < best_val:
                        best_val = val_loss
                        save_artifact(
                            {
                                "model_state": model.state_dict(),
                                "vocab": vocab,
                                "vocab_size": len(vocab["itos"]),
                                "config": config["model"],
                                "model_type": "bilstm",
                            },
                            config["checkpoint"]["local_path"],
                        )
                break
            except torch.OutOfMemoryError:
                if device == "cuda" and train_cfg["train"]["batch_size"] > 8:
                    train_cfg["train"]["batch_size"] = max(8, train_cfg["train"]["batch_size"] // 2)
                    torch.cuda.empty_cache()
                    print(f"[task2_bilstm] CUDA OOM, retrying with batch_size={train_cfg['train']['batch_size']}")
                    continue
                if device == "cuda":
                    device = "cpu"
                    train_cfg["train"]["use_cuda"] = False
                    print("[task2_bilstm] CUDA OOM persists, falling back to CPU")
                    continue
                raise
        maybe_push_to_hf(
            path=config["checkpoint"]["local_path"],
            enabled=config["checkpoint"].get("push_to_hf", False),
            repo_id=config["checkpoint"].get("repo_id"),
            filename=config["checkpoint"].get("hf_filename"),
        )

    model, _ = _load_model_from_artifact(config, device)
    _, val_loader, _ = build_mlm_dataloaders(config)
    val_loss, val_ppl = run_mlm_epoch(model, val_loader, None, device)
    out = {"task": "task2_bilstm", "val_loss": val_loss, "perplexity": val_ppl}
    if wandb_run is not None:
        log_wandb({"final_val_loss": val_loss, "final_val_perplexity": val_ppl})
    out_file = os.path.join(config["paths"]["results_dir"], "task2_bilstm.txt")
    with open(out_file, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(out, indent=2))
    print(f"[task2_bilstm] wrote results: {out_file}")
    if wandb_run is not None:
        finish_wandb()

from __future__ import annotations

import os

import torch

from src.common.artifacts import load_artifact, maybe_pull_from_hf, maybe_push_to_hf, save_artifact
from src.common.config import load_config
from src.common.data import read_lines, split_indices, subset_by_indices, validate_disjoint_splits
from src.common.io_utils import ensure_dir, pick_device, set_seed
from src.common.layers import ManualLSTMDecryptor
from src.task1.common import build_char_dataloaders, decrypt_lines, evaluate_metrics, run_epoch, write_results
from src.utils.hf_wandb import finish_wandb, init_wandb, log_wandb, save_wandb_run_info


def _build_model(model_config: dict, source_vocab_size: int, target_vocab_size: int) -> ManualLSTMDecryptor:
    return ManualLSTMDecryptor(
        vocab_size=source_vocab_size,
        output_vocab_size=target_vocab_size,
        emb_dim=model_config["emb_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config.get("num_layers", 1),
        dropout=model_config.get("dropout", 0.0),
    )


def _artifact_vocab_sizes(artifact: dict) -> tuple[int, int]:
    vocab = artifact["vocab"]
    if "source" in vocab and "target" in vocab:
        source_size = artifact.get("source_vocab_size", len(vocab["source"]["itos"]))
        target_size = artifact.get("target_vocab_size", len(vocab["target"]["itos"]))
        return source_size, target_size
    vocab_size = artifact["vocab_size"]
    return vocab_size, vocab_size


def _load_model_from_artifact(config: dict, device: str):
    model_path = maybe_pull_from_hf(
        source=config["checkpoint"]["source"],
        local_path=config["checkpoint"]["local_path"],
        repo_id=config["checkpoint"].get("repo_id"),
        filename=config["checkpoint"].get("hf_filename"),
    )
    artifact = load_artifact(model_path, device=device)
    source_vocab_size, target_vocab_size = _artifact_vocab_sizes(artifact)
    model = _build_model(artifact["config"], source_vocab_size, target_vocab_size).to(device)
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
            name=wandb_cfg.get("run_name", "task1_lstm"),
            dir=wandb_cfg.get("dir"),
        )

    ensure_dir(config["paths"]["results_dir"])
    ensure_dir(os.path.dirname(config["checkpoint"]["local_path"]))
    if wandb_run is not None:
        save_wandb_run_info(wandb_run, os.path.join(config["paths"]["results_dir"], "task1_lstm_wandb.json"))

    if mode in {"train", "both"}:
        train_loader, val_loader, vocab = build_char_dataloaders(config)
        model = _build_model(
            config["model"],
            len(vocab["source"]["itos"]),
            len(vocab["target"]["itos"]),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
        pad_idx = vocab["target"]["stoi"]["<pad>"]
        loss_name = config["train"].get("loss", "cross_entropy")
        grad_clip = config["train"].get("grad_clip")

        best_val = float("inf")
        for epoch in range(1, config["train"]["epochs"] + 1):
            train_loss = run_epoch(
                model,
                train_loader,
                optimizer,
                pad_idx,
                device,
                loss_name=loss_name,
                grad_clip=grad_clip,
            )
            val_loss = run_epoch(model, val_loader, None, pad_idx, device, loss_name=loss_name)
            print(f"[task1_lstm] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if wandb_run is not None:
                log_wandb({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            if val_loss < best_val:
                best_val = val_loss
                payload = {
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                    "source_vocab_size": len(vocab["source"]["itos"]),
                    "target_vocab_size": len(vocab["target"]["itos"]),
                    "config": config["model"],
                    "model_type": "lstm",
                }
                save_artifact(payload, config["checkpoint"]["local_path"])
        maybe_push_to_hf(
            path=config["checkpoint"]["local_path"],
            enabled=config["checkpoint"].get("push_to_hf", False),
            repo_id=config["checkpoint"].get("repo_id"),
            filename=config["checkpoint"].get("hf_filename"),
        )

    model, vocab = _load_model_from_artifact(config, device)
    cipher_eval = read_lines(config["data"]["cipher_eval_path"])
    plain = read_lines(config["data"]["plain_path"])
    usable = min(len(cipher_eval), len(plain))
    splits = split_indices(
        total_size=usable,
        val_ratio=config["data"].get("val_ratio", 0.1),
        test_ratio=config["data"].get("test_ratio", 0.0),
        seed=config["seed"],
    )
    validate_disjoint_splits(splits)
    selected = splits["test"] if config["data"].get("test_ratio", 0.0) > 0.0 else list(range(usable))
    cipher_eval = subset_by_indices(cipher_eval[:usable], selected)
    refs = subset_by_indices(plain[:usable], selected)
    decrypted = decrypt_lines(model, cipher_eval, vocab, device, config["data"].get("max_seq_len"))
    metrics = evaluate_metrics(decrypted, refs)
    if wandb_run is not None:
        log_wandb(metrics)
    out_file = os.path.join(config["paths"]["results_dir"], "task1_lstm.txt")
    write_results(out_file, metrics, decrypted)
    print(f"[task1_lstm] wrote results: {out_file}")
    if wandb_run is not None:
        finish_wandb()

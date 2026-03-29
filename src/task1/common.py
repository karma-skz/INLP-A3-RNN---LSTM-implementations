from __future__ import annotations

import json
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.data import (
    CIPHER_SPACE_TOKEN,
    CipherPlainDataset,
    build_char_vocab,
    build_cipher_vocab,
    collate_parallel_sequences,
    decode_chars,
    read_lines,
    split_indices,
    subset_by_indices,
    tokenize_ciphertext,
    validate_disjoint_splits,
)
from src.common.metrics import summarize_decryption_metrics


def _split_task1_vocab(vocab: dict) -> tuple[dict, dict]:
    if "source" in vocab and "target" in vocab:
        return vocab["source"], vocab["target"]
    return vocab, vocab


def build_char_dataloaders(config: dict):
    plain = read_lines(config["data"]["plain_path"])
    cipher = read_lines(config["data"]["cipher_train_path"])
    usable = min(len(plain), len(cipher))
    pairs = list(zip(cipher[:usable], plain[:usable]))

    splits = split_indices(
        total_size=len(pairs),
        val_ratio=config["data"].get("val_ratio", 0.1),
        test_ratio=config["data"].get("test_ratio", 0.0),
        seed=config["seed"],
    )
    validate_disjoint_splits(splits)
    train_pairs = subset_by_indices(pairs, splits["train"])
    val_pairs = subset_by_indices(pairs, splits["val"])

    source_vocab = build_cipher_vocab()
    source_pad_idx = source_vocab["stoi"]["<pad>"]

    # Build the plaintext vocabulary only from train data to avoid leakage.
    train_plain = [p for _, p in train_pairs]
    target_vocab = build_char_vocab(train_plain)
    target_pad_idx = target_vocab["stoi"]["<pad>"]

    max_len = config["data"].get("max_seq_len")
    train_ds = CipherPlainDataset(train_pairs, source_vocab["stoi"], target_vocab["stoi"], max_len=max_len)
    val_ds = CipherPlainDataset(val_pairs, source_vocab["stoi"], target_vocab["stoi"], max_len=max_len)

    collate = partial(
        collate_parallel_sequences,
        source_pad_idx=source_pad_idx,
        target_pad_idx=target_pad_idx,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )
    vocab = {
        "source": source_vocab,
        "target": target_vocab,
        "space_token": CIPHER_SPACE_TOKEN,
    }
    return train_loader, val_loader, vocab


def _compute_loss(logits: torch.Tensor, y: torch.Tensor, pad_idx: int, loss_name: str) -> torch.Tensor:
    # Ensure batch/time dimensions align between logits and targets.
    # logits expected shape: (B, T_out, V)
    # y expected shape: (B, T)
    if logits.dim() == 3 and y.dim() == 2:
        T_out = logits.size(1)
        T_tgt = y.size(1)
        if T_out != T_tgt:
            T = min(T_out, T_tgt)
            logits = logits[:, :T, :].contiguous()
            y = y[:, :T].contiguous()

    if loss_name == "l2":
        probs = torch.softmax(logits, dim=-1)
        # targets -> one-hot
        V = logits.size(-1)
        y_clamped = y.clamp(min=0)
        one_hot = torch.nn.functional.one_hot(y_clamped, num_classes=V).float().to(logits.device)
        mask = (y != pad_idx).unsqueeze(-1).to(logits.device)
        sq_err = (probs - one_hot).pow(2) * mask
        denom = mask.sum().clamp_min(1.0)
        return sq_err.sum() / denom

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    return criterion(logits.view(-1, logits.size(-1)), y.view(-1))


def run_epoch(
    model,
    loader,
    optimizer,
    pad_idx,
    device,
    loss_name: str = "cross_entropy",
    grad_clip: float | None = None,
):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    pbar = tqdm(loader, leave=False)
    for x, y, _lengths in pbar:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = _compute_loss(logits, y, pad_idx, loss_name)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def predict_loader(model, loader, vocab, device):
    model.eval()
    _source_vocab, target_vocab = _split_task1_vocab(vocab)
    pad_idx = target_vocab["stoi"]["<pad>"]
    preds, refs = [], []
    for x, y, lengths in loader:
        x = x.to(device)
        logits = model(x)
        pred_ids = logits.argmax(dim=-1).cpu()
        y_ids = y.cpu()
        for pred, ref, length in zip(pred_ids, y_ids, lengths.tolist()):
            preds.append(decode_chars(pred[:length].tolist(), target_vocab["itos"], pad_idx))
            refs.append(decode_chars(ref[:length].tolist(), target_vocab["itos"], pad_idx))
    return preds, refs


@torch.no_grad()
def decrypt_lines(
    model,
    lines: list[str],
    vocab: dict,
    device: str,
    max_len: int | None = None,
    batch_size: int = 256,
) -> list[str]:
    model.eval()
    source_vocab, target_vocab = _split_task1_vocab(vocab)
    stoi = source_vocab["stoi"]
    itos = target_vocab["itos"]
    target_pad_idx = target_vocab["stoi"]["<pad>"]
    out: list[str] = []
    for i in range(0, len(lines), batch_size):
        chunk = lines[i : i + batch_size]
        tensors = []
        lengths = []
        for line in chunk:
            tokens = tokenize_ciphertext(line)
            if max_len is not None:
                tokens = tokens[:max_len]
            ids = [stoi.get(token, stoi["<unk>"]) for token in tokens]
            tensors.append(torch.tensor(ids, dtype=torch.long))
            lengths.append(len(ids))
        x = torch.nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True,
            padding_value=stoi["<pad>"],
        ).to(device)
        logits = model(x)
        pred_ids = logits.argmax(dim=-1).cpu()
        for pred, length in zip(pred_ids, lengths):
            out.append(decode_chars(pred[:length].tolist(), itos, target_pad_idx))
    return out


def write_results(path: str, metrics: dict, preds: list[str] | None = None) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, indent=2))
        handle.write("\n")
        if preds is not None:
            handle.write("\n".join(preds))


def evaluate_metrics(preds: list[str], refs: list[str]) -> dict:
    return summarize_decryption_metrics(preds, refs)

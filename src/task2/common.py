from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.data import (
    MaskedLMDataset,
    NextWordDataset,
    build_word_vocab,
    collate_mlm,
    encode_words,
    read_lines,
    split_indices,
    subset_by_indices,
    tokenize_words,
    validate_disjoint_splits,
)


def prepare_word_data(config: dict):
    lines = read_lines(config["data"]["plain_path"])
    token_lists = [tokenize_words(line) for line in lines]
    token_lists = [toks for toks in token_lists if toks]
    splits = split_indices(
        total_size=len(token_lists),
        val_ratio=config["data"].get("val_ratio", 0.1),
        test_ratio=config["data"].get("test_ratio", 0.0),
        seed=config["seed"],
    )
    validate_disjoint_splits(splits)
    train = subset_by_indices(token_lists, splits["train"])
    val = subset_by_indices(token_lists, splits["val"])
    vocab = build_word_vocab(train, min_freq=config["data"].get("min_freq", 1))
    return train, val, vocab


def _flatten_nwp_tokens(token_lists: list[list[str]], vocab: dict, context_len: int) -> list[int]:
    bos = "<bos>"
    eos = "<eos>"
    flat: list[str] = []
    for row in token_lists:
        flat.extend([bos] * context_len)
        flat.extend(row)
        flat.append(eos)
    return encode_words(flat, vocab["stoi"])


def build_nwp_dataloaders(config: dict):
    train, val, vocab = prepare_word_data(config)
    context_len = config["model"]["context_len"]
    train_ids = _flatten_nwp_tokens(train, vocab, context_len)
    val_ids = _flatten_nwp_tokens(val, vocab, context_len)

    train_ds = NextWordDataset(
        train_ids,
        context_len=context_len,
        max_examples=config["data"].get("max_train_examples"),
    )
    val_ds = NextWordDataset(
        val_ids,
        context_len=context_len,
        max_examples=config["data"].get("max_val_examples"),
    )
    train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["train"]["batch_size"], shuffle=False)
    return train_loader, val_loader, vocab


def build_mlm_dataloaders(config: dict):
    train, val, vocab = prepare_word_data(config)
    max_train_rows = config["data"].get("max_train_rows")
    max_val_rows = config["data"].get("max_val_rows")
    if max_train_rows is not None:
        train = train[:max_train_rows]
    if max_val_rows is not None:
        val = val[:max_val_rows]
    pad = vocab["stoi"]["<pad>"]
    train_ds = MaskedLMDataset(
        train,
        stoi=vocab["stoi"],
        mask_prob=config["data"].get("mask_prob", 0.15),
        max_len=config["model"].get("max_len", 48),
    )
    val_ds = MaskedLMDataset(
        val,
        stoi=vocab["stoi"],
        mask_prob=config["data"].get("mask_prob", 0.15),
        max_len=config["model"].get("max_len", 48),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_mlm(b, pad),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_mlm(b, pad),
    )
    return train_loader, val_loader, vocab


def run_nwp_epoch(model, loader, optimizer, device, grad_clip: float | None = None):
    criterion = torch.nn.CrossEntropyLoss()
    train = optimizer is not None
    model.train(train)
    total = 0.0
    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total += loss.item()
    avg = total / max(len(loader), 1)
    return avg, math.exp(min(avg, 20.0))


def run_mlm_epoch(model, loader, optimizer, device, grad_clip: float | None = None):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    train = optimizer is not None
    model.train(train)
    total = 0.0
    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total += loss.item()
    avg = total / max(len(loader), 1)
    return avg, math.exp(min(avg, 20.0))

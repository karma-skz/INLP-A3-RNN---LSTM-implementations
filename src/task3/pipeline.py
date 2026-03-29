from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

import torch
from tqdm import tqdm

from src.common.artifacts import load_artifact, maybe_pull_from_hf
from src.common.config import load_config
from src.common.data import (
    decode_chars,
    encode_words,
    read_lines,
    split_indices,
    subset_by_indices,
    tokenize_ciphertext,
    tokenize_words,
    validate_disjoint_splits,
)
from src.common.io_utils import ensure_dir, pick_device, set_seed
from src.common.layers import ManualBiLSTMLM, ManualLSTMDecryptor, ManualRNNDecryptor, SimpleSSM
from src.common.metrics import summarize_decryption_metrics
from src.utils.hf_wandb import finish_wandb, init_wandb, log_wandb, save_wandb_run_info


@dataclass
class LexiconResources:
    words: list[str]
    word_counts: Counter
    gram_index: dict[str, set[int]]
    length_index: dict[int, list[int]]
    candidate_cache: dict[str, list[tuple[str, float]]]


def _split_decrypt_vocab(vocab: dict) -> tuple[dict, dict]:
    if "source" in vocab and "target" in vocab:
        return vocab["source"], vocab["target"]
    return vocab, vocab


def _artifact_vocab_sizes(artifact: dict) -> tuple[int, int]:
    vocab = artifact["vocab"]
    if "source" in vocab and "target" in vocab:
        source_size = artifact.get("source_vocab_size", len(vocab["source"]["itos"]))
        target_size = artifact.get("target_vocab_size", len(vocab["target"]["itos"]))
        return source_size, target_size
    vocab_size = artifact["vocab_size"]
    return vocab_size, vocab_size


def _load_decrypt_model(config: dict, device: str):
    ckpt = config["decryption"]["checkpoint"]
    path = maybe_pull_from_hf(
        source=ckpt["source"],
        local_path=ckpt["local_path"],
        repo_id=ckpt.get("repo_id"),
        filename=ckpt.get("hf_filename"),
    )
    artifact = load_artifact(path, device=device)
    model_cfg = artifact["config"]
    source_vocab_size, target_vocab_size = _artifact_vocab_sizes(artifact)
    model_type = artifact.get("model_type", config["decryption"].get("model_type", "lstm"))

    if model_type == "rnn":
        model = ManualRNNDecryptor(
            source_vocab_size,
            model_cfg["emb_dim"],
            model_cfg["hidden_dim"],
            output_vocab_size=target_vocab_size,
            num_layers=model_cfg.get("num_layers", 1),
            dropout=model_cfg.get("dropout", 0.0),
        )
    else:
        model = ManualLSTMDecryptor(
            source_vocab_size,
            model_cfg["emb_dim"],
            model_cfg["hidden_dim"],
            output_vocab_size=target_vocab_size,
            num_layers=model_cfg.get("num_layers", 1),
            dropout=model_cfg.get("dropout", 0.0),
        )
    model.load_state_dict(artifact["model_state"])
    model.to(device).eval()
    return model, artifact["vocab"]


def _load_lm_model(config: dict, device: str):
    ckpt = config["language_model"]["checkpoint"]
    path = maybe_pull_from_hf(
        source=ckpt["source"],
        local_path=ckpt["local_path"],
        repo_id=ckpt.get("repo_id"),
        filename=ckpt.get("hf_filename"),
    )
    artifact = load_artifact(path, device=device)
    model_cfg = artifact["config"]
    vocab_size = artifact["vocab_size"]
    model_type = artifact["model_type"]
    if model_type == "ssm":
        model = SimpleSSM(
            vocab_size,
            model_cfg["emb_dim"],
            model_cfg["state_dim"],
            num_layers=model_cfg.get("num_layers", 1),
            dropout=model_cfg.get("dropout", 0.0),
            tie_embeddings=model_cfg.get("tie_embeddings", False),
        )
    elif model_type == "bilstm":
        model = ManualBiLSTMLM(
            vocab_size,
            model_cfg["emb_dim"],
            model_cfg["hidden_dim"],
            num_layers=model_cfg.get("num_layers", 1),
            dropout=model_cfg.get("dropout", 0.0),
            tie_embeddings=model_cfg.get("tie_embeddings", False),
        )
    else:
        raise ValueError(f"Unsupported LM model type: {model_type}")
    model.load_state_dict(artifact["model_state"])
    model.to(device).eval()
    return model, artifact["vocab"], model_type


def _special_token_ids(stoi: dict[str, int]) -> set[int]:
    return {
        idx
        for token, idx in stoi.items()
        if token.startswith("<") and token.endswith(">")
    }


def _best_non_special(probs: torch.Tensor, blocked_ids: set[int]) -> tuple[int, float]:
    filtered = probs.clone()
    if blocked_ids:
        filtered[list(blocked_ids)] = 0.0
    best_idx = int(torch.argmax(filtered).item())
    return best_idx, float(filtered[best_idx].item())


def _char_ngrams(word: str) -> set[str]:
    padded = f"^{word}$"
    grams = set()
    for n in (2, 3):
        if len(padded) < n:
            continue
        for idx in range(len(padded) - n + 1):
            grams.add(padded[idx : idx + n])
    return grams or {word}


def _build_lexicon_resources(config: dict, lm_vocab: dict) -> LexiconResources:
    lines = read_lines(config["data"]["plain_path"])
    token_lists = [tokenize_words(line) for line in lines]
    token_lists = [tokens for tokens in token_lists if tokens]
    splits = split_indices(
        total_size=len(token_lists),
        val_ratio=config["data"].get("val_ratio", 0.1),
        test_ratio=config["data"].get("test_ratio", 0.0),
        seed=config["seed"],
    )
    validate_disjoint_splits(splits)
    train = subset_by_indices(token_lists, splits["train"])

    blocked = _special_token_ids(lm_vocab["stoi"])
    words = [
        word
        for idx, word in enumerate(lm_vocab["itos"])
        if idx not in blocked
    ]
    word_counts = Counter(tok for row in train for tok in row if tok in lm_vocab["stoi"])
    gram_index: dict[str, set[int]] = defaultdict(set)
    length_index: dict[int, list[int]] = defaultdict(list)
    for idx, word in enumerate(words):
        length_index[len(word)].append(idx)
        for gram in _char_ngrams(word):
            gram_index[gram].add(idx)
    return LexiconResources(
        words=words,
        word_counts=word_counts,
        gram_index=dict(gram_index),
        length_index=dict(length_index),
        candidate_cache={},
    )


def _is_repairable_token(token: str) -> bool:
    return len(token) >= 4 and any(ch.isalpha() for ch in token)


def _candidate_words(word: str, resources: LexiconResources, limit: int = 8) -> list[tuple[str, float]]:
    word = word.lower()
    cached = resources.candidate_cache.get(word)
    if cached is not None:
        return cached
    if not _is_repairable_token(word):
        resources.candidate_cache[word] = []
        return []

    grams = _char_ngrams(word)
    max_len_delta = 2 if len(word) <= 6 else 3 if len(word) <= 10 else 4
    hit_counts: Counter = Counter()
    for gram in grams:
        for idx in resources.gram_index.get(gram, ()):
            cand = resources.words[idx]
            if abs(len(cand) - len(word)) <= max_len_delta:
                hit_counts[idx] += 1

    if not hit_counts:
        for cand_len in range(max(1, len(word) - 1), len(word) + 2):
            for idx in resources.length_index.get(cand_len, ()):
                hit_counts[idx] += 0

    scored: list[tuple[float, str, float]] = []
    for idx, overlap in hit_counts.most_common(96):
        cand = resources.words[idx]
        ratio = SequenceMatcher(None, word, cand).ratio()
        if ratio < 0.45:
            continue
        gram_score = overlap / max(len(grams), 1)
        score = ratio + 0.2 * gram_score + 0.03 * math.log(resources.word_counts.get(cand, 0) + 1)
        if cand[0] == word[0]:
            score += 0.03
        if cand[-1] == word[-1]:
            score += 0.03
        scored.append((score, cand, ratio))

    scored.sort(reverse=True)
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for _score, cand, ratio in scored:
        if cand in seen:
            continue
        seen.add(cand)
        out.append((cand, ratio))
        if len(out) >= limit:
            break
    resources.candidate_cache[word] = out
    return out


def _match_surface_case(source: str, candidate: str) -> str:
    if source.isupper():
        return candidate.upper()
    if source.istitle():
        return candidate.title()
    return candidate


def _candidate_score(prob: float, similarity: float, frequency: int) -> float:
    freq_boost = 1.0 + 0.03 * math.log(frequency + 1)
    sim_boost = 0.55 + 0.45 * similarity
    return prob * sim_boost * freq_boost


def _pick_replacement(
    token: str,
    surface_token: str,
    current_id: int,
    current_prob: float,
    probs: torch.Tensor,
    stoi: dict[str, int],
    resources: LexiconResources,
    unk_id: int,
    threshold: float,
) -> str | None:
    if current_id != unk_id:
        return None

    candidates = _candidate_words(token, resources, limit=3)
    if not candidates:
        return None

    ranked: list[tuple[float, str, float, float]] = []

    for cand, similarity in candidates:
        if similarity < 0.68:
            continue
        if abs(len(cand) - len(token)) > 1:
            continue
        cand_id = stoi.get(cand)
        if cand_id is None or cand_id >= probs.numel():
            continue
        cand_prob = float(probs[cand_id].item())
        cand_score = _candidate_score(cand_prob, similarity, resources.word_counts.get(cand, 0))
        ranked.append((cand_score, cand, cand_prob, similarity))

    ranked.sort(reverse=True)
    if not ranked:
        return None

    best_score, best_word, best_prob, best_similarity = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0.0

    if best_similarity >= 0.8 and best_prob >= 1e-5:
        if runner_up > 0.0 and best_score <= runner_up * 1.5:
            return None
        return _match_surface_case(surface_token, best_word)

    if best_similarity >= 0.7 and best_prob >= 5e-4:
        if runner_up > 0.0 and best_score <= runner_up * 1.5:
            return None
        return _match_surface_case(surface_token, best_word)

    if best_prob < 1e-5:
        return None
    if runner_up > 0.0 and best_score <= runner_up * 1.5:
        return None
    return None


def _should_replace_token(
    current_id: int,
    current_prob: float,
    candidate_id: int,
    candidate_prob: float,
    unk_id: int,
    threshold: float,
) -> bool:
    if candidate_id == current_id:
        return False
    # Preserve OOV words from the decryptor output; mapping them to <unk> in the
    # LM should not give the LM permission to hallucinate replacements.
    if current_id == unk_id:
        return False
    replace_threshold = max(threshold, 0.5)
    return candidate_prob >= replace_threshold and candidate_prob > max(current_prob * 10.0, current_prob + 0.4)


@torch.no_grad()
def _decrypt_lines(
    model,
    vocab: dict,
    lines: list[str],
    device: str,
    max_len: int | None,
    batch_size: int = 256,
    progress_desc: str | None = None,
):
    source_vocab, target_vocab = _split_decrypt_vocab(vocab)
    stoi = source_vocab["stoi"]
    itos = target_vocab["itos"]
    target_pad_idx = target_vocab["stoi"]["<pad>"]
    outputs = []
    iterator = range(0, len(lines), batch_size)
    if progress_desc is not None:
        iterator = tqdm(iterator, desc=progress_desc, leave=False)
    for i in iterator:
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
            outputs.append(decode_chars(pred[:length].tolist(), itos, target_pad_idx))
    return outputs


@torch.no_grad()
def _correct_with_ssm(model, lm_vocab: dict, lines: list[str], context_len: int, threshold: float, device: str):
    stoi = lm_vocab["stoi"]
    itos = lm_vocab["itos"]
    unk = stoi["<unk>"]
    bos = stoi.get("<bos>")
    blocked_ids = _special_token_ids(stoi)
    corrected = []
    for line in lines:
        surface_tokens = [w for w in line.strip().split() if w]
        tokens = tokenize_words(line)
        if not tokens:
            corrected.append("")
            continue
        if len(surface_tokens) != len(tokens):
            surface_tokens = tokens.copy()
        if bos is None and len(tokens) <= context_len:
            corrected.append(" ".join(surface_tokens))
            continue
        ids = encode_words(tokens, stoi)
        out_ids = ids.copy()
        out_tokens = surface_tokens.copy()
        if bos is None:
            for i in range(context_len, len(ids)):
                ctx = torch.tensor(out_ids[i - context_len : i], dtype=torch.long, device=device).unsqueeze(0)
                logits = model(ctx)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                cur = out_ids[i]
                cur_prob = probs[cur].item() if cur < probs.numel() else 0.0
                cand, cand_prob = _best_non_special(probs, blocked_ids)
                if _should_replace_token(cur, cur_prob, cand, cand_prob, unk, threshold):
                    out_ids[i] = cand
                    out_tokens[i] = itos[cand]
        else:
            history = [bos] * context_len + out_ids.copy()
            for i in range(len(ids)):
                ctx = torch.tensor(history[i : i + context_len], dtype=torch.long, device=device).unsqueeze(0)
                logits = model(ctx)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                cur = out_ids[i]
                cur_prob = probs[cur].item() if cur < probs.numel() else 0.0
                cand, cand_prob = _best_non_special(probs, blocked_ids)
                if _should_replace_token(cur, cur_prob, cand, cand_prob, unk, threshold):
                    out_ids[i] = cand
                    out_tokens[i] = itos[cand]
                    history[context_len + i] = out_ids[i]
        corrected.append(" ".join(out_tokens))
    return corrected


@torch.no_grad()
def _correct_with_ssm_lexicon(
    model,
    lm_vocab: dict,
    lines: list[str],
    context_len: int,
    threshold: float,
    device: str,
    resources: LexiconResources,
    progress_desc: str | None = None,
):
    stoi = lm_vocab["stoi"]
    unk = stoi["<unk>"]
    bos = stoi.get("<bos>")
    corrected = []
    line_iter = lines
    if progress_desc is not None:
        line_iter = tqdm(lines, desc=progress_desc, leave=False)
    for line in line_iter:
        surface_tokens = [w for w in line.strip().split() if w]
        tokens = tokenize_words(line)
        if not tokens:
            corrected.append("")
            continue
        if len(surface_tokens) != len(tokens):
            surface_tokens = tokens.copy()
        ids = encode_words(tokens, stoi)
        out_ids = ids.copy()
        out_tokens = surface_tokens.copy()
        history = ([bos] * context_len if bos is not None else []) + out_ids.copy()
        start = 0 if bos is not None else context_len
        for i in range(start, len(ids)):
            if out_ids[i] != unk or not _is_repairable_token(tokens[i]):
                continue
            if bos is not None:
                ctx_ids = history[i : i + context_len]
            else:
                ctx_ids = out_ids[i - context_len : i]
            if len(ctx_ids) < context_len:
                continue
            ctx = torch.tensor(ctx_ids, dtype=torch.long, device=device).unsqueeze(0)
            probs = torch.softmax(model(ctx), dim=-1).squeeze(0)
            cur = out_ids[i]
            cur_prob = probs[cur].item() if cur < probs.numel() else 0.0
            replacement = _pick_replacement(
                tokens[i],
                out_tokens[i],
                cur,
                cur_prob,
                probs,
                stoi,
                resources,
                unk,
                threshold,
            )
            if replacement is None:
                continue
            replacement_lc = replacement.lower()
            repl_id = stoi.get(replacement_lc)
            if repl_id is None:
                continue
            out_ids[i] = repl_id
            out_tokens[i] = replacement
            if bos is not None:
                history[context_len + i] = repl_id
        corrected.append(" ".join(out_tokens))
    return corrected


@torch.no_grad()
def _correct_with_bilstm(model, lm_vocab: dict, lines: list[str], threshold: float, device: str):
    stoi = lm_vocab["stoi"]
    itos = lm_vocab["itos"]
    mask = stoi["<mask>"]
    unk = stoi["<unk>"]
    blocked_ids = _special_token_ids(stoi)
    corrected = []
    for line in lines:
        surface_tokens = [w for w in line.strip().split() if w]
        tokens = tokenize_words(line)
        if not tokens:
            corrected.append("")
            continue
        if len(surface_tokens) != len(tokens):
            surface_tokens = tokens.copy()
        ids = encode_words(tokens, stoi)
        out_ids = ids.copy()
        out_tokens = surface_tokens.copy()
        for i in range(len(ids)):
            masked = out_ids.copy()
            masked[i] = mask
            x = torch.tensor(masked, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x).squeeze(0)
            probs = torch.softmax(logits[i], dim=-1)
            cur = out_ids[i]
            cur_prob = probs[cur].item() if cur < probs.numel() else 0.0
            cand, cand_prob = _best_non_special(probs, blocked_ids)
            if _should_replace_token(cur, cur_prob, cand, cand_prob, unk, threshold):
                out_ids[i] = cand
                out_tokens[i] = itos[cand]
        corrected.append(" ".join(out_tokens))
    return corrected


@torch.no_grad()
def _correct_with_bilstm_lexicon(
    model,
    lm_vocab: dict,
    lines: list[str],
    threshold: float,
    device: str,
    resources: LexiconResources,
    progress_desc: str | None = None,
):
    stoi = lm_vocab["stoi"]
    mask = stoi["<mask>"]
    unk = stoi["<unk>"]
    corrected = []
    line_iter = lines
    if progress_desc is not None:
        line_iter = tqdm(lines, desc=progress_desc, leave=False)
    for line in line_iter:
        surface_tokens = [w for w in line.strip().split() if w]
        tokens = tokenize_words(line)
        if not tokens:
            corrected.append("")
            continue
        if len(surface_tokens) != len(tokens):
            surface_tokens = tokens.copy()
        ids = encode_words(tokens, stoi)
        out_ids = ids.copy()
        out_tokens = surface_tokens.copy()
        for i in range(len(ids)):
            token = tokens[i]
            if out_ids[i] != unk or not _is_repairable_token(token):
                continue
            masked = out_ids.copy()
            masked[i] = mask
            x = torch.tensor(masked, dtype=torch.long, device=device).unsqueeze(0)
            probs = torch.softmax(model(x).squeeze(0)[i], dim=-1)
            cur = out_ids[i]
            cur_prob = probs[cur].item() if cur < probs.numel() else 0.0
            replacement = _pick_replacement(
                token,
                out_tokens[i],
                cur,
                cur_prob,
                probs,
                stoi,
                resources,
                unk,
                threshold,
            )
            if replacement is None:
                continue
            replacement_lc = replacement.lower()
            repl_id = stoi.get(replacement_lc)
            if repl_id is None:
                continue
            out_ids[i] = repl_id
            out_tokens[i] = replacement
        corrected.append(" ".join(out_tokens))
    return corrected


def _write_text(path: str, lines: list[str]):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _write_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2))


@torch.no_grad()
def _perplexity_ssm(
    model,
    lm_vocab: dict,
    lines: list[str],
    context_len: int,
    device: str,
    progress_desc: str | None = None,
) -> float:
    stoi = lm_vocab["stoi"]
    bos = stoi.get("<bos>")
    eps = 1e-12
    total_nll = 0.0
    total_tokens = 0
    line_iter = lines
    if progress_desc is not None:
        line_iter = tqdm(lines, desc=progress_desc, leave=False)
    for line in line_iter:
        ids = encode_words(tokenize_words(line), stoi)
        if bos is None and len(ids) <= context_len:
            continue
        if bos is None:
            for i in range(context_len, len(ids)):
                ctx = torch.tensor(ids[i - context_len : i], dtype=torch.long, device=device).unsqueeze(0)
                logits = model(ctx)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                tgt = ids[i]
                prob = probs[tgt].item() if tgt < probs.numel() else 0.0
                total_nll += -math.log(max(prob, eps))
                total_tokens += 1
        else:
            history = [bos] * context_len + ids
            for i, tgt in enumerate(ids):
                ctx = torch.tensor(history[i : i + context_len], dtype=torch.long, device=device).unsqueeze(0)
                logits = model(ctx)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                prob = probs[tgt].item() if tgt < probs.numel() else 0.0
                total_nll += -math.log(max(prob, eps))
                total_tokens += 1
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


@torch.no_grad()
def _perplexity_bilstm(model, lm_vocab: dict, lines: list[str], device: str, progress_desc: str | None = None) -> float:
    stoi = lm_vocab["stoi"]
    mask = stoi["<mask>"]
    eps = 1e-12
    total_nll = 0.0
    total_tokens = 0
    line_iter = lines
    if progress_desc is not None:
        line_iter = tqdm(lines, desc=progress_desc, leave=False)
    for line in line_iter:
        ids = encode_words(tokenize_words(line), stoi)
        if not ids:
            continue
        for i in range(len(ids)):
            masked = ids.copy()
            masked[i] = mask
            x = torch.tensor(masked, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x).squeeze(0)
            probs = torch.softmax(logits[i], dim=-1)
            tgt = ids[i]
            prob = probs[tgt].item() if tgt < probs.numel() else 0.0
            total_nll += -math.log(max(prob, eps))
            total_tokens += 1
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def main(config_path: str, mode: str = "evaluate"):
    del mode  # Task 3 primarily evaluates by loading existing checkpoints.
    config = load_config(config_path)
    set_seed(config["seed"])
    device = pick_device(config.get("use_cuda", True))
    wandb_cfg = config.get("wandb", {})
    wandb_run = None
    if wandb_cfg.get("enabled", False):
        wandb_run = init_wandb(
            project=wandb_cfg.get("project", "inlp-a3"),
            config=config,
            name=wandb_cfg.get("run_name", "task3"),
            dir=wandb_cfg.get("dir"),
        )
    ensure_dir(config["paths"]["results_dir"])
    if wandb_run is not None:
        save_wandb_run_info(
            wandb_run,
            os.path.join(config["paths"]["results_dir"], f"task3_{config['wandb'].get('run_name', 'task3')}_wandb.json"),
        )

    decrypt_model, decrypt_vocab = _load_decrypt_model(config, device)
    lm_model, lm_vocab, lm_type = _load_lm_model(config, device)
    lexicon_resources = _build_lexicon_resources(config, lm_vocab)

    plain = read_lines(config["data"]["plain_path"])
    out_name = "task3_ssm.txt" if lm_type == "ssm" else "task3_bilstm.txt"
    summary_file = os.path.join(config["paths"]["results_dir"], out_name)
    metrics_report = {
        "lm_type": lm_type,
        "progress": {
            "status": "running",
            "completed_ciphers": 0,
            "total_ciphers": len(config["data"]["cipher_paths"]),
            "current_cipher": None,
        },
        "noise_levels": {},
    }
    _write_json(summary_file, metrics_report)

    cipher_iter = tqdm(config["data"]["cipher_paths"], desc=f"task3_{lm_type}", leave=True)
    for cipher_idx, cipher_path in enumerate(cipher_iter, start=1):
        noise_name = os.path.splitext(os.path.basename(cipher_path))[0]
        metrics_report["progress"]["current_cipher"] = noise_name
        _write_json(summary_file, metrics_report)
        cipher_lines = read_lines(cipher_path)
        usable = min(len(cipher_lines), len(plain))
        splits = split_indices(
            total_size=usable,
            val_ratio=config["data"].get("val_ratio", 0.1),
            test_ratio=config["data"].get("test_ratio", 0.0),
            seed=config["seed"],
        )
        validate_disjoint_splits(splits)

        split_name = config["data"].get("eval_split", "test")
        if split_name not in {"train", "val", "test"}:
            raise ValueError("data.eval_split must be one of: train, val, test")

        if split_name == "test" and config["data"].get("test_ratio", 0.0) == 0.0:
            selected = list(range(usable))
        else:
            selected = splits[split_name]

        cipher_lines = subset_by_indices(cipher_lines[:usable], selected)
        refs = subset_by_indices(plain[:usable], selected)

        limit = config["data"].get("max_lines")
        if limit is not None:
            cipher_lines = cipher_lines[:limit]
            refs = refs[:limit]

        base_pred = _decrypt_lines(
            decrypt_model,
            decrypt_vocab,
            cipher_lines,
            device=device,
            max_len=config["decryption"].get("max_seq_len"),
            progress_desc=f"{noise_name}: decrypt",
        )

        if lm_type == "ssm":
            corrected = _correct_with_ssm_lexicon(
                lm_model,
                lm_vocab,
                base_pred,
                context_len=config["language_model"]["context_len"],
                threshold=config["language_model"]["replace_threshold"],
                device=device,
                resources=lexicon_resources,
                progress_desc=f"{noise_name}: repair",
            )
        else:
            corrected = _correct_with_bilstm_lexicon(
                lm_model,
                lm_vocab,
                base_pred,
                threshold=config["language_model"]["replace_threshold"],
                device=device,
                resources=lexicon_resources,
                progress_desc=f"{noise_name}: repair",
            )

        base_metrics = summarize_decryption_metrics(base_pred, refs)
        corrected_metrics = summarize_decryption_metrics(corrected, refs)

        if lm_type == "ssm":
            base_ppl = _perplexity_ssm(
                lm_model,
                lm_vocab,
                base_pred,
                context_len=config["language_model"]["context_len"],
                device=device,
                progress_desc=f"{noise_name}: base ppl",
            )
            corrected_ppl = _perplexity_ssm(
                lm_model,
                lm_vocab,
                corrected,
                context_len=config["language_model"]["context_len"],
                device=device,
                progress_desc=f"{noise_name}: corr ppl",
            )
        else:
            base_ppl = _perplexity_bilstm(
                lm_model,
                lm_vocab,
                base_pred,
                device=device,
                progress_desc=f"{noise_name}: base ppl",
            )
            corrected_ppl = _perplexity_bilstm(
                lm_model,
                lm_vocab,
                corrected,
                device=device,
                progress_desc=f"{noise_name}: corr ppl",
            )

        base_metrics["perplexity"] = base_ppl
        corrected_metrics["perplexity"] = corrected_ppl
        metrics_report["noise_levels"][noise_name] = {
            "decryption_only": base_metrics,
            f"decryption_plus_{lm_type}": corrected_metrics,
        }
        if wandb_run is not None:
            log_wandb(
                {
                    f"{noise_name}/decryption_only_char_accuracy": base_metrics["char_accuracy"],
                    f"{noise_name}/decryption_only_word_accuracy": base_metrics["word_accuracy"],
                    f"{noise_name}/decryption_only_levenshtein": base_metrics["avg_levenshtein"],
                    f"{noise_name}/decryption_only_perplexity": base_metrics["perplexity"],
                    f"{noise_name}/decryption_plus_{lm_type}_char_accuracy": corrected_metrics["char_accuracy"],
                    f"{noise_name}/decryption_plus_{lm_type}_word_accuracy": corrected_metrics["word_accuracy"],
                    f"{noise_name}/decryption_plus_{lm_type}_levenshtein": corrected_metrics["avg_levenshtein"],
                    f"{noise_name}/decryption_plus_{lm_type}_perplexity": corrected_metrics["perplexity"],
                }
            )

        base_file = os.path.join(config["paths"]["results_dir"], f"{noise_name}_decryption_only.txt")
        corr_file = os.path.join(config["paths"]["results_dir"], f"{noise_name}_decryption_plus_{lm_type}.txt")
        _write_text(base_file, base_pred)
        _write_text(corr_file, corrected)
        metrics_report["progress"]["completed_ciphers"] = cipher_idx
        metrics_report["progress"]["current_cipher"] = None
        _write_json(summary_file, metrics_report)

    metrics_report["progress"]["status"] = "completed"
    _write_json(summary_file, metrics_report)
    print(f"[task3_{lm_type}] wrote results: {summary_file}")
    if wandb_run is not None:
        finish_wandb()

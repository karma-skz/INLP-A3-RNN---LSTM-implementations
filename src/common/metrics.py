from __future__ import annotations

import math
from collections import Counter


def char_accuracy(pred: str, ref: str) -> float:
    if not ref:
        return 1.0 if not pred else 0.0
    matches = sum(1 for p, r in zip(pred, ref) if p == r)
    return matches / max(len(ref), 1)


def word_accuracy(pred: str, ref: str) -> float:
    p = pred.split()
    r = ref.split()
    if not r:
        return 1.0 if not p else 0.0
    matches = sum(1 for pw, rw in zip(p, r) if pw == rw)
    return matches / max(len(r), 1)


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def corpus_bleu1(preds: list[str], refs: list[str]) -> float:
    clipped = 0
    total = 0
    pred_len = 0
    ref_len = 0
    for pred, ref in zip(preds, refs):
        p_tokens = pred.split()
        r_tokens = ref.split()
        pred_len += len(p_tokens)
        ref_len += len(r_tokens)
        p_counts = Counter(p_tokens)
        r_counts = Counter(r_tokens)
        clipped += sum(min(c, r_counts[w]) for w, c in p_counts.items())
        total += len(p_tokens)
    if total == 0:
        return 0.0
    precision = clipped / total
    if pred_len == 0:
        return 0.0
    bp = 1.0 if pred_len > ref_len else math.exp(1 - (ref_len / max(pred_len, 1)))
    return bp * precision


def _lcs_len(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    p = pred.split()
    r = ref.split()
    if not p or not r:
        return 0.0
    lcs = _lcs_len(p, r)
    prec = lcs / len(p)
    rec = lcs / len(r)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def summarize_decryption_metrics(preds: list[str], refs: list[str]) -> dict[str, float]:
    n = max(len(refs), 1)
    char_acc = sum(char_accuracy(p, r) for p, r in zip(preds, refs)) / n
    word_acc = sum(word_accuracy(p, r) for p, r in zip(preds, refs)) / n
    lev = sum(levenshtein_distance(p, r) for p, r in zip(preds, refs)) / n
    bleu1 = corpus_bleu1(preds, refs)
    rouge = sum(rouge_l_f1(p, r) for p, r in zip(preds, refs)) / n
    return {
        "char_accuracy": char_acc,
        "word_accuracy": word_acc,
        "avg_levenshtein": lev,
        "bleu1": bleu1,
        "rouge_l_f1": rouge,
    }

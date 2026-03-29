from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


SPECIAL_CHARS = ["<pad>", "<unk>"]
SPECIAL_WORDS = ["<pad>", "<unk>", "<mask>", "<bos>", "<eos>"]
CIPHER_SPACE_TOKEN = "9"


def read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def train_val_split(data: list, val_ratio: float, seed: int) -> tuple[list, list]:
    idx = list(range(len(data)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_size = int(len(idx) * val_ratio)
    val_idx = set(idx[:val_size])
    train, val = [], []
    for i, item in enumerate(data):
        (val if i in val_idx else train).append(item)
    return train, val


def split_indices(total_size: int, val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[int]]:
    if total_size < 0:
        raise ValueError("total_size must be non-negative")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError("test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    idx = list(range(total_size))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_test = int(total_size * test_ratio)
    n_val = int(total_size * val_ratio)
    test_idx = sorted(idx[:n_test])
    val_idx = sorted(idx[n_test : n_test + n_val])
    train_idx = sorted(idx[n_test + n_val :])
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def subset_by_indices(data: list, indices: list[int]) -> list:
    return [data[i] for i in indices]


def validate_disjoint_splits(splits: dict[str, list[int]]) -> None:
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    if train & val or train & test or val & test:
        raise ValueError("Data split leakage detected: train/val/test overlap")


def build_char_vocab(lines: list[str]) -> dict:
    charset = set()
    for line in lines:
        charset.update(line)
    itos = SPECIAL_CHARS + sorted(charset)
    stoi = {ch: i for i, ch in enumerate(itos)}
    return {"stoi": stoi, "itos": itos}


def encode_chars(text: str, stoi: dict[str, int]) -> list[int]:
    unk = stoi["<unk>"]
    return [stoi.get(ch, unk) for ch in text]


def decode_chars(ids: list[int], itos: list[str], pad_idx: int) -> str:
    out = []
    for idx in ids:
        if idx == pad_idx:
            continue
        token = itos[idx]
        if token.startswith("<") and token.endswith(">"):
            continue
        out.append(token)
    return "".join(out)


def tokenize_ciphertext(text: str, space_token: str = CIPHER_SPACE_TOKEN) -> list[str]:
    tokens = []
    pos = 0
    while pos < len(text):
        if text[pos] == space_token:
            tokens.append(space_token)
            pos += 1
            continue
        if pos + 1 >= len(text):
            raise ValueError(f"Ciphertext has a dangling digit at position {pos}: {text!r}")
        tokens.append(text[pos : pos + 2])
        pos += 2
    return tokens


def build_cipher_vocab() -> dict:
    # Task 1 ciphertext symbols are one dedicated space token plus all possible
    # two-digit symbols. Using the full symbol inventory avoids OOVs when the
    # trained decryptor is later applied to the other cipher files in Task 3.
    itos = SPECIAL_CHARS + [CIPHER_SPACE_TOKEN] + [f"{idx:02d}" for idx in range(100)]
    stoi = {token: i for i, token in enumerate(itos)}
    return {"stoi": stoi, "itos": itos}


def encode_tokens(tokens: list[str], stoi: dict[str, int]) -> list[int]:
    unk = stoi["<unk>"]
    return [stoi.get(token, unk) for token in tokens]


class CharPairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], stoi: dict[str, int], max_len: int | None = None) -> None:
        self.pairs = pairs
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        cipher, plain = self.pairs[idx]
        if self.max_len is not None:
            cipher = cipher[: self.max_len]
            plain = plain[: self.max_len]
        x = torch.tensor(encode_chars(cipher, self.stoi), dtype=torch.long)
        y = torch.tensor(encode_chars(plain, self.stoi), dtype=torch.long)
        return x, y


class CipherPlainDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        source_stoi: dict[str, int],
        target_stoi: dict[str, int],
        max_len: int | None = None,
    ) -> None:
        self.pairs = pairs
        self.source_stoi = source_stoi
        self.target_stoi = target_stoi
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        cipher, plain = self.pairs[idx]
        cipher_tokens = tokenize_ciphertext(cipher)
        plain_chars = list(plain)
        if len(cipher_tokens) != len(plain_chars):
            raise ValueError(
                "Cipher/plain lengths do not align after tokenization: "
                f"{len(cipher_tokens)} vs {len(plain_chars)}"
            )
        if self.max_len is not None:
            cipher_tokens = cipher_tokens[: self.max_len]
            plain_chars = plain_chars[: self.max_len]
        x = torch.tensor(encode_tokens(cipher_tokens, self.source_stoi), dtype=torch.long)
        y = torch.tensor(encode_chars("".join(plain_chars), self.target_stoi), dtype=torch.long)
        return x, y, len(plain_chars)


def collate_char_pairs(batch, pad_idx: int):
    xs, ys = zip(*batch)
    xpad = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_idx)
    ypad = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=pad_idx)
    return xpad, ypad


def collate_parallel_sequences(batch, source_pad_idx: int, target_pad_idx: int):
    xs, ys, lengths = zip(*batch)
    xpad = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=source_pad_idx)
    ypad = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=target_pad_idx)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return xpad, ypad, lengths


def tokenize_words(line: str) -> list[str]:
    return [w for w in line.lower().strip().split() if w]


def build_word_vocab(token_lists: list[list[str]], min_freq: int = 1) -> dict:
    freq = {}
    for tokens in token_lists:
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1
    words = [tok for tok, c in freq.items() if c >= min_freq]
    itos = SPECIAL_WORDS + sorted(words)
    stoi = {w: i for i, w in enumerate(itos)}
    return {"stoi": stoi, "itos": itos}


def encode_words(tokens: list[str], stoi: dict[str, int]) -> list[int]:
    unk = stoi["<unk>"]
    return [stoi.get(tok, unk) for tok in tokens]


def decode_words(ids: list[int], itos: list[str], pad_idx: int) -> list[str]:
    words = []
    for i in ids:
        if i == pad_idx:
            continue
        tok = itos[i]
        if tok.startswith("<") and tok.endswith(">"):
            continue
        words.append(tok)
    return words


class NextWordDataset(Dataset):
    def __init__(self, ids: list[int], context_len: int, max_examples: int | None = None):
        self.samples = []
        for i in range(context_len, len(ids)):
            x = ids[i - context_len : i]
            y = ids[i]
            self.samples.append((x, y))
            if max_examples is not None and len(self.samples) >= max_examples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class MaskedLMDataset(Dataset):
    def __init__(
        self,
        token_lists: list[list[str]],
        stoi: dict[str, int],
        mask_prob: float = 0.15,
        max_len: int = 48,
    ) -> None:
        self.rows = [encode_words(tokens[:max_len], stoi) for tokens in token_lists if len(tokens) > 1]
        self.mask_prob = mask_prob
        self.pad = stoi["<pad>"]
        self.mask = stoi["<mask>"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        ids = self.rows[idx]
        x = ids.copy()
        y = [-100] * len(ids)
        masked_any = False
        for i in range(len(ids)):
            if random.random() < self.mask_prob:
                y[i] = ids[i]
                x[i] = self.mask
                masked_any = True
        if not masked_any:
            pos = random.randrange(len(ids))
            y[pos] = ids[pos]
            x[pos] = self.mask
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_mlm(batch, pad_idx: int):
    xs, ys = zip(*batch)
    xpad = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_idx)
    ypad = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    return xpad, ypad


@dataclass
class TextPairs:
    train_pairs: list[tuple[str, str]]
    val_pairs: list[tuple[str, str]]

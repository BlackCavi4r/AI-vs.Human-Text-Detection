import math
import re
import string
from collections import Counter
from typing import Dict, Iterable, List

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: object) -> str:
    if text is None:
        return ""
    text = str(text)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\xa0": " ",
        "â€™": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "‚Äô": "'",
        "‚Äù": '"',
        "‚Äú": '"',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: object) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(clean_text(text))]


def split_sentences(text: object) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    return sentences if sentences else [text]


def count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    vowels = "aeiouy"
    groups = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            groups += 1
        prev_vowel = is_vowel
    if word.endswith("e") and groups > 1:
        groups -= 1
    return max(groups, 1)


def linguistic_feature_names() -> List[str]:
    return [
        "char_count",
        "word_count",
        "sentence_count",
        "avg_word_length",
        "avg_sentence_length",
        "vocab_richness",
        "punctuation_ratio",
        "comma_count",
        "period_count",
        "question_count",
        "exclamation_count",
        "semicolon_colon_count",
        "quote_count",
        "uppercase_ratio",
        "digit_ratio",
        "flesch_reading_ease",
    ]


def linguistic_features_one(text: object) -> List[float]:
    raw = clean_text(text)
    tokens = tokenize(raw)
    sentences = split_sentences(raw)

    char_count = len(raw)
    word_count = len(tokens)
    sentence_count = len(sentences)
    avg_word_length = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
    avg_sentence_length = word_count / sentence_count if sentence_count else 0.0
    vocab_richness = len(set(tokens)) / word_count if word_count else 0.0

    punct_count = sum(1 for ch in raw if ch in string.punctuation)
    punctuation_ratio = punct_count / char_count if char_count else 0.0
    comma_count = raw.count(",")
    period_count = raw.count(".")
    question_count = raw.count("?")
    exclamation_count = raw.count("!")
    semicolon_colon_count = raw.count(";") + raw.count(":")
    quote_count = raw.count("'") + raw.count('"')

    alpha_chars = [ch for ch in raw if ch.isalpha()]
    uppercase_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars) if alpha_chars else 0.0
    digit_ratio = sum(1 for ch in raw if ch.isdigit()) / char_count if char_count else 0.0

    syllables = sum(count_syllables(t) for t in tokens)
    if word_count and sentence_count:
        flesch = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllables / word_count)
    else:
        flesch = 0.0

    return [
        char_count,
        word_count,
        sentence_count,
        avg_word_length,
        avg_sentence_length,
        vocab_richness,
        punctuation_ratio,
        comma_count,
        period_count,
        question_count,
        exclamation_count,
        semicolon_colon_count,
        quote_count,
        uppercase_ratio,
        digit_ratio,
        flesch,
    ]


def compute_linguistic_features(texts: Iterable[object]) -> np.ndarray:
    return np.array([linguistic_features_one(t) for t in texts], dtype=np.float32)


def summarize_text_stats(text: object) -> Dict[str, float]:
    names = linguistic_feature_names()
    values = linguistic_features_one(text)
    return dict(zip(names, values))


def build_vocab(texts: Iterable[object], max_vocab_size: int = 15000, min_freq: int = 2) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common(max_vocab_size - 2):
        if freq < min_freq:
            continue
        vocab[word] = len(vocab)
    return vocab


def texts_to_sequences(texts: Iterable[object], vocab: Dict[str, int], max_len: int = 250) -> np.ndarray:
    sequences = []
    unk = vocab.get("<UNK>", 1)
    pad = vocab.get("<PAD>", 0)
    for text in texts:
        ids = [vocab.get(tok, unk) for tok in tokenize(text)[:max_len]]
        if len(ids) < max_len:
            ids.extend([pad] * (max_len - len(ids)))
        sequences.append(ids)
    return np.array(sequences, dtype=np.int64)


def average_word_vectors(tokenized_texts: Iterable[List[str]], keyed_vectors, vector_size: int) -> np.ndarray:
    rows = []
    for tokens in tokenized_texts:
        vecs = [keyed_vectors[w] for w in tokens if w in keyed_vectors]
        if vecs:
            rows.append(np.mean(vecs, axis=0))
        else:
            rows.append(np.zeros(vector_size, dtype=np.float32))
    return np.array(rows, dtype=np.float32)


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


class FNNTextClassifier(__import__("torch").nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.3, pad_idx: int = 0):
        torch, nn, F = _torch()
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.pad_idx = pad_idx

    def forward(self, x):
        torch, nn, F = _torch()
        emb = self.embedding(x)
        mask = (x != self.pad_idx).unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        hidden = F.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)
        return self.fc2(hidden).squeeze(1)


class LSTMTextClassifier(__import__("torch").nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 64, dropout: float = 0.3, pad_idx: int = 0):
        torch, nn, F = _torch()
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        hidden = self.dropout(h_n[-1])
        return self.fc(hidden).squeeze(1)


class CNNTextClassifier(__import__("torch").nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, num_filters: int = 64, filter_sizes=(3, 4, 5), dropout: float = 0.3, pad_idx: int = 0):
        torch, nn, F = _torch()
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, k) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, x):
        torch, nn, F = _torch()
        emb = self.embedding(x).transpose(1, 2)
        conv_outs = [F.relu(conv(emb)) for conv in self.convs]
        pooled = [F.max_pool1d(out, out.shape[2]).squeeze(2) for out in conv_outs]
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        return self.fc(cat).squeeze(1)


def create_deep_model(model_name: str, vocab_size: int, config: Dict) -> object:
    name = model_name.lower()
    if name == "fnn":
        return FNNTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.get("embedding_dim", 64),
            hidden_dim=config.get("hidden_dim", 128),
            dropout=config.get("dropout", 0.3),
        )
    if name == "lstm":
        return LSTMTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.get("embedding_dim", 64),
            hidden_dim=config.get("hidden_dim", 64),
            dropout=config.get("dropout", 0.3),
        )
    if name == "cnn":
        return CNNTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.get("embedding_dim", 64),
            num_filters=config.get("num_filters", 64),
            filter_sizes=tuple(config.get("filter_sizes", (3, 4, 5))),
            dropout=config.get("dropout", 0.3),
        )
    raise ValueError(f"Unknown deep model name: {model_name}")

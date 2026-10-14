"""Model loading and batched scoring."""

import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(model_dir: Path | str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device()).eval()
    return tokenizer, model


@torch.inference_mode()
def predict_proba(
    texts: list[str], tokenizer, model, batch_size: int = 64, max_length: int = 128
) -> np.ndarray:
    """Probability of the `toxic` class for every text."""
    dev = next(model.parameters()).device
    # sort by length so that batches need less padding
    order = np.argsort([len(t) for t in texts])
    scores = np.empty(len(texts), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        idx = order[start : start + batch_size]
        batch = tokenizer(
            [texts[i] for i in idx],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(dev)
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            logits = model(**batch).logits
        scores[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return scores


def latency_ms(texts: list[str], tokenizer, model, n: int = 200, max_length: int = 128) -> float:
    """Average time to score one comment (batch size 1), in milliseconds."""
    sample = texts[:n]
    predict_proba(sample[:10], tokenizer, model, batch_size=1, max_length=max_length)  # warm-up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    predict_proba(sample, tokenizer, model, batch_size=1, max_length=max_length)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / len(sample) * 1000

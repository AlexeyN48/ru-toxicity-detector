"""Bias/fairness analysis over identity subgroups (no torch dependency).

Methodology: Borkan et al., 2019, "Nuanced Metrics for Measuring Unintended Bias with Real Data
for Text Classification" (Jigsaw Unintended Bias in Toxicity Classification).
"""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pymorphy3
import yaml
from sklearn.metrics import roc_auc_score

TERMS_PATH = Path("docs/fairness/identity_terms.yaml")
MIN_SUBGROUP_SIZE = 30

# Acceptance criteria from the Model Card, section 5
MAX_AUC_GAP = 0.05
MAX_FPR_GAP = 0.05
MAX_TEMPLATE_FPR = 0.10

_WORD = re.compile(r"[а-яёa-z]+", re.IGNORECASE)


@lru_cache(maxsize=1)
def _morph() -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer()


def load_terms(path: Path = TERMS_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=200_000)
def _lemma(word: str) -> str:
    return _morph().parse(word)[0].normal_form


def lemmas(text: str) -> set[str]:
    return {_lemma(w.lower()) for w in _WORD.findall(text)}


def subgroup_masks(texts: list[str], categories: dict) -> dict[str, np.ndarray]:
    """For every category: boolean mask of texts that mention any of its terms (by lemma)."""
    text_lemmas = [lemmas(t) for t in texts]
    masks = {}
    for name, cat in categories.items():
        terms = {_lemma(t) for t in cat["terms"]}
        masks[name] = np.array([bool(terms & tl) for tl in text_lemmas])
    return masks


def _auc(y: np.ndarray, s: np.ndarray) -> float | None:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def _rate(mask: np.ndarray) -> float | None:
    return float(mask.mean()) if mask.size else None


def bootstrap_ci(
    y: np.ndarray, s: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float] | None:
    """95 % bootstrap confidence interval of ROC-AUC."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            values.append(roc_auc_score(y[idx], s[idx]))
    if not values:
        return None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def subgroup_metrics(y: np.ndarray, scores: np.ndarray, mask: np.ndarray, threshold: float) -> dict:
    """Subgroup AUC, BPSN AUC, BNSP AUC, FPR and FNR for one subgroup."""
    pred = scores >= threshold
    sub, bg = mask, ~mask
    bpsn = (sub & (y == 0)) | (bg & (y == 1))
    bnsp = (sub & (y == 1)) | (bg & (y == 0))
    return {
        "n": int(sub.sum()),
        "n_toxic": int((sub & (y == 1)).sum()),
        "reliable": bool(sub.sum() >= MIN_SUBGROUP_SIZE and len(np.unique(y[sub])) == 2),
        "subgroup_auc": _auc(y[sub], scores[sub]),
        "subgroup_auc_ci95": bootstrap_ci(y[sub], scores[sub]) if sub.sum() >= 2 else None,
        "bpsn_auc": _auc(y[bpsn], scores[bpsn]),
        "bnsp_auc": _auc(y[bnsp], scores[bnsp]),
        "fpr": _rate(pred[sub & (y == 0)]),
        "fnr": _rate(~pred[sub & (y == 1)]),
    }


def template_sentences(categories: dict, templates: list[str]) -> list[tuple[str, str]]:
    """Neutral sentences built from templates: list of (category, sentence)."""
    out = []
    for name, cat in categories.items():
        for term in cat["terms"]:
            parsed = _morph().parse(term)[0]
            inflected = parsed.inflect({"ablt"})
            term_ins = inflected.word if inflected else term
            for tpl in templates:
                sentence = tpl.format(term=term, term_ins=term_ins)
                out.append((name, sentence[0].upper() + sentence[1:]))
    return out


def analyze(
    texts: list[str],
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    template_scores: dict[str, np.ndarray],
    terms: dict,
) -> dict:
    """Full fairness report with pass/fail against the Model Card criteria."""
    overall_auc = float(roc_auc_score(y, scores))
    pred = scores >= threshold
    overall_fpr = float(pred[y == 0].mean())
    masks = subgroup_masks(texts, terms["categories"])

    groups, failures = {}, []
    for name, mask in masks.items():
        m = subgroup_metrics(y, scores, mask, threshold)
        m["title"] = terms["categories"][name]["title"]
        m["template_fpr"] = float((template_scores[name] >= threshold).mean())
        checks = {"template_fpr": m["template_fpr"] <= MAX_TEMPLATE_FPR}
        if m["reliable"]:
            checks["subgroup_auc"] = m["subgroup_auc"] >= overall_auc - MAX_AUC_GAP
            checks["bpsn_auc"] = m["bpsn_auc"] >= overall_auc - MAX_AUC_GAP
            if m["fpr"] is not None:
                checks["fpr_gap"] = m["fpr"] - overall_fpr <= MAX_FPR_GAP
        m["checks"] = checks
        failures += [f"{name}.{c}" for c, ok in checks.items() if not ok]
        groups[name] = m

    return {
        "overall_roc_auc": overall_auc,
        "overall_fpr": overall_fpr,
        "threshold": threshold,
        "criteria": {
            "max_auc_gap": MAX_AUC_GAP,
            "max_fpr_gap": MAX_FPR_GAP,
            "max_template_fpr": MAX_TEMPLATE_FPR,
            "min_subgroup_size": MIN_SUBGROUP_SIZE,
        },
        "subgroups": groups,
        "failures": failures,
        "passed": not failures,
    }

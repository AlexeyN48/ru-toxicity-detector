import numpy as np

from toxicity.metrics import check_targets, choose_threshold, classification_report


def test_choose_threshold_keeps_required_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    scores = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 2000), 0, 1)
    threshold = choose_threshold(y, scores, min_recall=0.85)
    report = classification_report(y, scores, threshold)
    assert report["recall_toxic"] >= 0.85


def test_perfect_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    report = classification_report(y, scores, threshold=0.5)
    assert report["roc_auc"] == 1.0
    assert report["f1_toxic"] == 1.0
    assert all(check_targets(report).values())

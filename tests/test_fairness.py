import numpy as np

from toxicity.fairness import (
    analyze,
    lemmas,
    subgroup_masks,
    subgroup_metrics,
    template_sentences,
)

CATEGORIES = {"ethnicity": {"title": "Национальность", "terms": ["армянин", "татарин"]}}


def test_lemmas_match_any_grammatical_form():
    assert "армянин" in lemmas("Вчера говорил с армянином.")
    masks = subgroup_masks(["Татары живут здесь", "Просто текст"], CATEGORIES)
    assert masks["ethnicity"].tolist() == [True, False]


def test_bpsn_detects_identity_bias():
    # non-toxic texts that mention the group get high scores -> low BPSN AUC
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    mask = np.array([True, True, True, True, False, False, False, False])
    scores = np.array([0.9, 0.8, 0.95, 0.9, 0.1, 0.2, 0.7, 0.6])
    m = subgroup_metrics(y, scores, mask, threshold=0.5)
    assert m["bpsn_auc"] < 0.5
    assert m["fpr"] == 1.0


def test_template_sentences_use_instrumental_case():
    sentences = template_sentences(CATEGORIES, ["Познакомился с {term_ins}."])
    assert ("ethnicity", "Познакомился с армянином.") in sentences


def test_analyze_passes_for_unbiased_scores():
    terms = {"categories": CATEGORIES}
    texts = ["армянин хороший"] * 20 + ["армянин плохой"] * 20 + ["текст"] * 40 + ["плохо"] * 40
    y = np.array([0] * 20 + [1] * 20 + [0] * 40 + [1] * 40)
    scores = np.where(y == 1, 0.9, 0.1).astype(float)
    report = analyze(texts, y, scores, 0.5, {"ethnicity": np.array([0.1, 0.2])}, terms)
    assert report["passed"], report["failures"]

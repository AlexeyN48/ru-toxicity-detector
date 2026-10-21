from pathlib import Path

import yaml

from toxicity.metrics import TARGETS

POLICY = Path("configs/auto_retraining.yaml")


def load() -> dict:
    return yaml.safe_load(POLICY.read_text(encoding="utf-8"))


def test_quality_gates_match_model_card_targets():
    gates = load()["gates"]
    assert gates["roc_auc"] == TARGETS["roc_auc"]
    assert gates["f1_toxic"] == TARGETS["f1_toxic"]
    assert gates["fairness_passed"] is True


def test_one_learning_rate_per_attempt():
    attempts = load()["attempts"]
    assert len(attempts["learning_rates"]) == attempts["max_attempts"]
    assert all(lr > 0 for lr in attempts["learning_rates"])


def test_training_stop_criteria_are_consistent():
    training = load()["training"]
    assert Path(training["config"]).exists()
    assert 0 < training["early_stopping_patience"] < training["max_epochs"]
    assert training["time_budget_minutes"] > 0


def test_canary_is_a_small_share_of_traffic():
    canary = load()["canary"]
    assert 0 < canary["traffic_share"] <= 0.25
    assert canary["duration_hours"] >= 1

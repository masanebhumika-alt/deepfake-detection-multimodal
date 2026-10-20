"""Tests for the confidence-calibrated fusion layer (app/fusion.py)."""
import pytest

from app.fusion import fuse


def _r(modality, conf):
    return {"modality": modality, "confidence_ai_generated": conf}


def test_empty_input_returns_unknown():
    out = fuse([])
    assert out["label"] == "Unknown"
    assert out["fused_confidence"] is None


@pytest.mark.parametrize(
    "conf, label, tier",
    [
        (90, "Likely Fake", "High Confidence"),
        (65, "Likely Fake", "High Confidence"),      # threshold: >= 65
        (64.9, "Possibly Fake", "Low Confidence"),
        (50, "Possibly Fake", "Low Confidence"),     # threshold: >= 50
        (49.9, "Possibly Real", "Low Confidence"),
        (35, "Possibly Real", "Low Confidence"),     # threshold: >= 35
        (34.9, "Likely Real", "High Confidence"),
        (5, "Likely Real", "High Confidence"),
    ],
)
def test_single_modality_thresholds(conf, label, tier):
    out = fuse([_r("text", conf)])
    assert out["label"] == label
    assert out["confidence_tier"] == tier
    # with one modality the fused score equals that modality's score
    assert out["fused_confidence_ai_generated"] == pytest.approx(conf, abs=0.01)


def test_more_certain_modality_gets_more_weight():
    out = fuse([_r("text", 95), _r("audio", 52)])
    w = out["modality_weights"]
    assert w["text"] > w["audio"]


def test_weights_sum_to_one():
    out = fuse([_r("text", 90), _r("image", 20), _r("audio", 50)])
    assert sum(out["modality_weights"].values()) == pytest.approx(1.0, abs=0.01)


def test_floor_weight_means_no_modality_ignored():
    # A fully uncertain modality (50) still gets non-zero weight.
    out = fuse([_r("text", 90), _r("audio", 50)])
    assert out["modality_weights"]["audio"] > 0


def test_fused_value_matches_hand_calculation():
    results = [_r("text", 90), _r("audio", 50)]
    w_text = 0.15 + 0.85 * (abs(90 - 50) / 50)
    w_audio = 0.15 + 0.85 * 0
    expected = (w_text * 90 + w_audio * 50) / (w_text + w_audio)
    out = fuse(results)
    assert out["fused_confidence_ai_generated"] == pytest.approx(expected, abs=0.01)


def test_per_modality_results_are_returned():
    results = [_r("text", 80), _r("image", 30)]
    out = fuse(results)
    assert out["per_modality"] == results

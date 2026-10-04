import os
import sys

from app.models.audio_detector import FEATURE_ORDER, predict

SAMPLE = sys.argv[1] if len(sys.argv) > 1 else "data/audio/mini_asv/flac/sample.flac"

if not os.path.exists(SAMPLE):
    sys.exit(f"[FAIL] sample not found: {SAMPLE}")

result = predict(SAMPLE)

print(f"scorer                   : {result['scorer']}")
print(f"model_path_used          : {result['model_path']}")
print(f"feature_order_expected   : {FEATURE_ORDER}")
print(f"feature_values           : {result['features']}")
print(f"confidence_ai_generated  : {result['confidence_ai_generated']}")
print(f"label                    : {result['label']}")

assert result["scorer"] == "trained_model", (
    f"[FAIL] scorer={result['scorer']} - fallback still active. "
    "Check artifacts exist at app/models/artifacts/audio_model.joblib or artifacts/audio_model.joblib"
)
assert len(result["features"]) == 3, f"[FAIL] expected 3 features, got {len(result['features'])}"
assert list(FEATURE_ORDER) == [
    "spectral_flatness_mean",
    "zero_crossing_rate_std",
    "mfcc_variance_mean",
], f"[FAIL] feature order mismatch: {FEATURE_ORDER}"
assert 0.0 <= result["confidence_ai_generated"] <= 100.0, "[FAIL] confidence out of 0-100 range"

print("[PASS] trained_model active, features aligned, confidence in range")

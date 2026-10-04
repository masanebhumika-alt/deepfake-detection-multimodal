import os

import joblib
import librosa
import numpy as np


FEATURE_ORDER = ["spectral_flatness_mean", "zero_crossing_rate_std", "mfcc_variance_mean"]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATHS = [
    os.path.join(_THIS_DIR, "artifacts", "audio_model.joblib"),
    os.path.join(_THIS_DIR, "..", "..", "artifacts", "audio_model.joblib"),
]

_model = None
_model_path_used = None
for _path in _MODEL_PATHS:
    if os.path.exists(_path):
        _model = joblib.load(_path)
        _model_path_used = _path
        break


def extract_features(filepath: str) -> np.ndarray:
    y, sr = librosa.load(filepath, sr=None, mono=True)

    spectral_flatness_mean = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    zero_crossing_rate_std = float(np.std(librosa.feature.zero_crossing_rate(y=y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_variance_mean = float(np.mean(np.var(mfcc, axis=1)))

    return np.array(
        [spectral_flatness_mean, zero_crossing_rate_std, mfcc_variance_mean],
        dtype=float,
    )


def _heuristic_score(features: np.ndarray) -> float:
    spectral_flatness, zero_crossing_rate, mfcc_variance = features
    raw = (spectral_flatness * 2.0) - (zero_crossing_rate * 1.5) - (mfcc_variance * 0.001)
    return float(1.0 / (1.0 + np.exp(-raw)))


def predict(filepath: str) -> dict:
    features = extract_features(filepath)

    if _model is not None:
        probabilities = _model.predict_proba(features.reshape(1, -1))[0]
        spoof_probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        scorer = "trained_model"
    else:
        spoof_probability = _heuristic_score(features)
        scorer = "heuristic_fallback"

    confidence = round(spoof_probability * 100, 2)
    return {
        "modality": "audio",
        "scorer": scorer,
        "model_path": _model_path_used,
        "features": features.tolist(),
        "confidence_ai_generated": confidence,
        "label": "ai_generated" if spoof_probability >= 0.5 else "human",
    }

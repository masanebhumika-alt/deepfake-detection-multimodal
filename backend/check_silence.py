import json
from pathlib import Path

import librosa
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score

import joblib
from app.models.audio_detector import FEATURE_ORDER
from compare_models import cached, chunked, wilson_interval


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "app" / "models" / "artifacts"
TOP_DB = 30


def extract_trimmed_features(paths):
    rows = []
    for path in paths:
        waveform, sample_rate = librosa.load(str(path), sr=None, mono=True)
        trimmed, _ = librosa.effects.trim(waveform, top_db=TOP_DB)
        spectral_flatness_mean = float(np.mean(librosa.feature.spectral_flatness(y=trimmed)))
        zero_crossing_rate_std = float(np.std(librosa.feature.zero_crossing_rate(y=trimmed)))
        mfcc = librosa.feature.mfcc(y=trimmed, sr=sample_rate, n_mfcc=13)
        mfcc_variance_mean = float(np.mean(np.var(mfcc, axis=1)))
        rows.append([
            spectral_flatness_mean,
            zero_crossing_rate_std,
            mfcc_variance_mean,
        ])
    return np.asarray(rows, dtype=float)


def main():
    manifest_path = ARTIFACTS / "audio_model.split.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_entries = manifest["train"]
    test_entries = manifest["test"]
    entries = train_entries + test_entries
    paths = [Path(entry["file"]) for entry in entries]
    labels = np.asarray([entry["label"] for entry in entries], dtype=int)

    features, cached_labels = cached(
        f"audio_silence_trim_top_db_{TOP_DB}",
        paths,
        labels,
        lambda: chunked(
            f"audio_silence_trim_top_db_{TOP_DB}_chunk",
            paths,
            labels,
            extract_trimmed_features,
        ),
    )
    if not np.array_equal(cached_labels, labels):
        raise ValueError("Cached trimmed-audio labels do not match the split manifest")
    if features.shape[1] != len(FEATURE_ORDER):
        raise ValueError(
            f"Expected {len(FEATURE_ORDER)} audio features, got {features.shape[1]}"
        )

    train_count = len(train_entries)
    test_labels = labels[train_count:]
    train_features = features[:train_count]
    test_features = features[train_count:]
    production_model = joblib.load(ARTIFACTS / "audio_model.joblib")
    classifier = clone(production_model)
    classifier.fit(train_features, labels[:train_count])
    predictions = classifier.predict(test_features)
    positive_index = list(classifier.classes_).index(1)
    positive_scores = classifier.predict_proba(test_features)[:, positive_index]
    correct = int(np.count_nonzero(predictions == test_labels))

    report = {
        "dataset": "mini_asv",
        "preprocessing": f"librosa.effects.trim(top_db={TOP_DB}) before feature extraction",
        "features": FEATURE_ORDER,
        "split": {
            "source": "audio_model.split.json",
            "random_state": manifest.get("random_state"),
            "test_size": manifest.get("test_size"),
            "train_count": int(train_count),
            "test_count": int(len(test_labels)),
            "train_class_counts": {
                str(label): int(np.count_nonzero(labels[:train_count] == label))
                for label in np.unique(labels[:train_count])
            },
            "test_class_counts": {
                str(label): int(np.count_nonzero(test_labels == label))
                for label in np.unique(test_labels)
            },
        },
        "model": {
            "type": type(classifier).__name__,
            "parameters": classifier.get_params(),
        },
        "metrics": {
            "accuracy": float(accuracy_score(test_labels, predictions)),
            "accuracy_ci_95_wilson": wilson_interval(correct, len(test_labels)),
            "auc": float(roc_auc_score(test_labels, positive_scores)),
        },
        "feature_cache": f"training_logs/cache/audio_silence_trim_top_db_{TOP_DB}.npz; reused only on matching fingerprint",
    }

    output_path = ROOT / "training_logs" / "silence_check.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    main()
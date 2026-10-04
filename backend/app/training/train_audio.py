import os
import sys
import argparse
import hashlib
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from app.models.audio_detector import extract_features, FEATURE_ORDER

LABEL_MAP = {"bonafide": 0, "spoof": 1}


def parse_protocol(protocol_path: str, audio_dir: str, ext: str = ".flac"):
    rows = []
    with open(protocol_path) as protocol_file:
        for line_number, line in enumerate(protocol_file, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[SKIP] line {line_number}: bad column count ({len(parts)}) -> {parts}")
                continue
            _, filename, _, _, key = parts
            if key not in LABEL_MAP:
                print(f"[SKIP] line {line_number}: unknown label '{key}'")
                continue
            file_path = os.path.join(audio_dir, filename + ext)
            if not os.path.exists(file_path):
                print(f"[SKIP] line {line_number}: missing file {file_path}")
                continue
            rows.append((file_path, LABEL_MAP[key]))
    return rows


def build_dataset(rows):
    features, labels = [], []
    used_rows = []
    for index, (file_path, label) in enumerate(rows, 1):
        try:
            feature_vector = extract_features(file_path)
            features.append(feature_vector)
            labels.append(label)
            used_rows.append((file_path, label))
        except Exception as error:
            print(f"[WARN] feature extraction failed for {file_path}: {error}")
        if index % 100 == 0:
            print(f"  ...{index}/{len(rows)} processed")
    return np.asarray(features, dtype=float), np.asarray(labels), used_rows


def save_artifact(model, split_manifest):
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    artifact_path = os.path.join(backend_root, "app", "models", "artifacts", "audio_model.joblib")
    manifest_path = os.path.join(backend_root, "app", "models", "artifacts", "audio_model.split.json")
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(model, artifact_path)
    split_manifest["model_path"] = os.path.abspath(artifact_path)
    with open(artifact_path, "rb") as artifact_file:
        split_manifest["model_sha256"] = hashlib.sha256(artifact_file.read()).hexdigest()
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(split_manifest, manifest_file, indent=2)
    print(f"[SAVED] {artifact_path}")
    print(f"[SAVED] {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol_path")
    parser.add_argument("audio_dir")
    parser.add_argument("--ext", default=".flac")
    args = parser.parse_args()

    print(f"[INFO] FEATURE_ORDER = {FEATURE_ORDER}")
    rows = parse_protocol(args.protocol_path, args.audio_dir, args.ext)
    print(f"[INFO] {len(rows)} usable rows after parsing")
    if len(rows) < 10:
        sys.exit("[FAIL] too few usable rows - check protocol/audio_dir alignment")

    features, labels, used_rows = build_dataset(rows)
    print(f"[INFO] dataset shape X={features.shape} y={labels.shape} class_balance={np.bincount(labels)}")

    train_indices, test_indices = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42
    )
    train_features, test_features = features[train_indices], features[test_indices]
    train_labels, test_labels = labels[train_indices], labels[test_indices]

    classifier = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    classifier.fit(train_features, train_labels)

    print(classification_report(test_labels, classifier.predict(test_features), target_names=["bonafide", "spoof"]))
    with open(args.protocol_path, "rb") as protocol_file:
        protocol_sha256 = hashlib.sha256(protocol_file.read()).hexdigest()
    split_manifest = {
        "protocol_path": os.path.abspath(args.protocol_path),
        "protocol_sha256": protocol_sha256,
        "audio_dir": os.path.abspath(args.audio_dir),
        "random_state": 42,
        "test_size": 0.2,
        "train": [
            {"file": os.path.abspath(used_rows[index][0]), "label": used_rows[index][1]}
            for index in train_indices
        ],
        "test": [
            {"file": os.path.abspath(used_rows[index][0]), "label": used_rows[index][1]}
            for index in test_indices
        ],
    }
    save_artifact(classifier, split_manifest)


if __name__ == "__main__":
    main()

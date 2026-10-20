import hashlib
import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack

from app.training import train_text


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "app" / "models" / "artifacts"
TEXT_SAMPLES_PER_CLASS = train_text.SAMPLES_PER_CLASS
RANDOM_STATE = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _label_counts(labels):
    values, counts = np.unique(labels, return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def grouped_text_evaluation():
    csv_path = ROOT / "hc3_flat.csv"
    frame = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError(f"Expected text and label columns in {csv_path}")

    expanded_texts = []
    expanded_labels = []
    source_answer_ids = []
    for source_answer_id, (text, label) in enumerate(
        zip(frame["text"].astype(str), frame["label"].astype(int))
    ):
        if label not in (0, 1):
            continue
        chunks = train_text.chunk_text_for_short_samples(text)
        expanded_texts.extend(chunks)
        expanded_labels.extend([label] * len(chunks))
        source_answer_ids.extend([source_answer_id] * len(chunks))

    expanded_labels = np.asarray(expanded_labels, dtype=int)
    source_answer_ids = np.asarray(source_answer_ids, dtype=int)
    class_indices = {
        label: np.flatnonzero(expanded_labels == label)
        for label in (0, 1)
    }
    samples_per_class = min(
        TEXT_SAMPLES_PER_CLASS,
        *(len(indices) for indices in class_indices.values()),
    )
    if samples_per_class == 0:
        raise ValueError("HC3 data must contain examples for both labels 0 and 1")

    rng = np.random.default_rng(train_text.SAMPLING_RANDOM_STATE)
    selected_indices = np.concatenate([
        rng.choice(class_indices[label], size=samples_per_class, replace=False)
        for label in (0, 1)
    ])
    rng.shuffle(selected_indices)

    texts = np.asarray(expanded_texts, dtype=object)[selected_indices]
    labels = expanded_labels[selected_indices]
    groups = source_answer_ids[selected_indices]
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    train_indices, test_indices = next(splitter.split(texts, labels, groups))
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    if len(np.unique(train_labels)) != 2 or len(np.unique(test_labels)) != 2:
        raise ValueError("Grouped split did not retain both labels in train and test")

    train_texts = texts[train_indices].tolist()
    test_texts = texts[test_indices].tolist()
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=1000, analyzer="word"
    )
    char_vectorizer = TfidfVectorizer(
        ngram_range=(3, 5), max_features=1000, analyzer="char"
    )
    word_train = word_vectorizer.fit_transform(train_texts)
    word_test = word_vectorizer.transform(test_texts)
    char_train = char_vectorizer.fit_transform(train_texts)
    char_test = char_vectorizer.transform(test_texts)

    all_custom_features = train_text.custom_feature_matrix(
        texts.tolist(), "grouped leakage check"
    )
    scaler = StandardScaler()
    custom_train = scaler.fit_transform(all_custom_features[train_indices])
    custom_test = scaler.transform(all_custom_features[test_indices])
    train_features = hstack([
        csr_matrix(custom_train), word_train, char_train
    ]).tocsr()
    test_features = hstack([
        csr_matrix(custom_test), word_test, char_test
    ]).tocsr()

    artifact = joblib.load(ARTIFACTS / "text_model.joblib")
    estimator = clone(artifact["classifier"])
    estimator.fit(train_features, train_labels)
    predictions = estimator.predict(test_features)

    train_groups = set(groups[train_indices].tolist())
    test_groups = set(groups[test_indices].tolist())
    train_text_values = set(train_texts)
    test_text_values = set(test_texts)
    return {
        "dataset": "HC3 flattened answer rows",
        "csv_path": str(csv_path.relative_to(ROOT)),
        "source_group_key": "zero-based CSV row index; local CSV has no source-answer ID column",
        "split_method": "GroupShuffleSplit(test_size=0.2, random_state=42)",
        "selected_per_class": int(samples_per_class),
        "selected_chunks": int(len(labels)),
        "train_chunks": int(len(train_indices)),
        "test_chunks": int(len(test_indices)),
        "train_label_counts": _label_counts(train_labels),
        "test_label_counts": _label_counts(test_labels),
        "train_source_answers": int(len(train_groups)),
        "test_source_answers": int(len(test_groups)),
        "source_answers_in_both_splits": int(len(train_groups & test_groups)),
        "exact_text_values_in_both_splits": int(len(train_text_values & test_text_values)),
        "production_estimator_type": type(artifact["classifier"]).__name__,
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "f1": float(f1_score(test_labels, predictions, pos_label=1, zero_division=0)),
    }


def _md5_file(path):
    digest = hashlib.md5()
    with path.open("rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_duplicate_report():
    image_root = ROOT / "data" / "faces" / "real_vs_fake" / "real-vs-fake"
    splits = ("train", "valid", "test")
    hashes_by_split = {}
    image_counts = {}

    for split in splits:
        split_root = image_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Image split folder not found: {split_root}")
        hashes = {}
        image_paths = sorted(
            path for path in split_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        image_counts[split] = len(image_paths)
        for path in image_paths:
            hashes.setdefault(_md5_file(path), []).append(
                str(path.relative_to(ROOT))
            )
        hashes_by_split[split] = hashes

    duplicate_counts_by_pair = {}
    for first, second in combinations(splits, 2):
        shared_hashes = set(hashes_by_split[first]) & set(hashes_by_split[second])
        duplicate_counts_by_pair[f"{first}/{second}"] = {
            "shared_md5_hashes": int(len(shared_hashes)),
            "matching_file_pairs": int(sum(
                len(hashes_by_split[first][digest])
                * len(hashes_by_split[second][digest])
                for digest in shared_hashes
            )),
        }

    all_hashes = set().union(*(set(value) for value in hashes_by_split.values()))
    cross_split_hashes = [
        digest for digest in all_hashes
        if sum(digest in hashes_by_split[split] for split in splits) > 1
    ]
    duplicate_copies = sum(
        sum(len(hashes_by_split[split].get(digest, [])) for split in splits) - 1
        for digest in cross_split_hashes
    )
    return {
        "dataset": "140k Real and Fake Faces",
        "image_root": str(image_root.relative_to(ROOT)),
        "hash_algorithm": "MD5",
        "image_files_scanned_by_split": image_counts,
        "total_image_files_scanned": int(sum(image_counts.values())),
        "unique_content_hashes_shared_across_splits": int(len(cross_split_hashes)),
        "duplicate_file_copies_across_splits": int(duplicate_copies),
        "duplicates_by_split_pair": duplicate_counts_by_pair,
    }


def main():
    report = {
        "text": grouped_text_evaluation(),
        "images": image_duplicate_report(),
    }
    output_path = ROOT / "training_logs" / "leakage_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.write("\n")
    print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    main()
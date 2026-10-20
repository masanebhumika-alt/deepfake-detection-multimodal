import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from app.training import train_text
from compare_models import cached_text_features, wilson_interval


ROOT = Path(__file__).resolve().parent
DATASET_ID = "Hello-SimpleAI/HC3"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def _answer_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def flatten_hc3_with_question_ids(records):
    texts = []
    labels = []
    question_ids = []

    for record_index, record in enumerate(records):
        question = re.sub(r"\s+", " ", str(record.get("question", ""))).strip().casefold()
        if not question:
            raise ValueError(f"HC3 record {record_index} has no question text")

        for answer in _answer_list(record.get("human_answers")):
            for chunk in train_text.chunk_text_for_short_samples(str(answer)):
                texts.append(chunk)
                labels.append(0)
                question_ids.append(question)

        for answer in _answer_list(record.get("chatgpt_answers")):
            for chunk in train_text.chunk_text_for_short_samples(str(answer)):
                texts.append(chunk)
                labels.append(1)
                question_ids.append(question)

    return (
        np.asarray(texts, dtype=object),
        np.asarray(labels, dtype=int),
        np.asarray(question_ids, dtype=object),
    )


def main():
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Install the optional 'datasets' package to reload HC3 from Hugging Face."
        ) from error

    records = load_dataset(DATASET_ID, "all")["train"]
    texts, labels, question_ids = flatten_hc3_with_question_ids(records)
    class_indices = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    samples_per_class = min(
        train_text.SAMPLES_PER_CLASS,
        *(len(indices) for indices in class_indices.values()),
    )
    if samples_per_class == 0:
        raise ValueError("HC3 data must contain answers for both labels")

    rng = np.random.default_rng(train_text.SAMPLING_RANDOM_STATE)
    selected = np.concatenate([
        rng.choice(class_indices[label], size=samples_per_class, replace=False)
        for label in (0, 1)
    ])
    rng.shuffle(selected)
    texts = texts[selected]
    labels = labels[selected]
    question_ids = question_ids[selected]

    split = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_indices, test_indices = next(split.split(texts, labels, question_ids))
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    if len(np.unique(train_labels)) != 2 or len(np.unique(test_labels)) != 2:
        raise ValueError("Question-grouped split must contain both labels in train and test")

    custom_features, cached_labels = cached_text_features(
        "text_gpt2_original_sample",
        texts,
        labels,
    )
    if not np.array_equal(cached_labels, labels):
        raise ValueError("Cached text labels do not match the selected HC3 rows")

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

    scaler = StandardScaler()
    custom_train = scaler.fit_transform(custom_features[train_indices])
    custom_test = scaler.transform(custom_features[test_indices])
    train_features = hstack([
        csr_matrix(custom_train), word_train, char_train
    ]).tocsr()
    test_features = hstack([
        csr_matrix(custom_test), word_test, char_test
    ]).tocsr()

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    probabilities = classifier.predict_proba(test_features)[:, list(classifier.classes_).index(1)]
    correct = int(np.count_nonzero(predictions == test_labels))

    train_group_ids = set(question_ids[train_indices].tolist())
    test_group_ids = set(question_ids[test_indices].tolist())
    report = {
        "dataset": DATASET_ID,
        "source_question_id": "normalized question text; HC3 rows expose question but no separate question ID",
        "chunking": "train_text.chunk_text_for_short_samples (up to three sentences)",
        "balanced_samples_per_class": int(samples_per_class),
        "selected_chunks": int(len(labels)),
        "split": {
            "method": "GroupShuffleSplit",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "sampling_seed": train_text.SAMPLING_RANDOM_STATE,
            "train_count": int(len(train_indices)),
            "test_count": int(len(test_indices)),
            "train_class_counts": {
                str(label): int(np.count_nonzero(train_labels == label))
                for label in np.unique(train_labels)
            },
            "test_class_counts": {
                str(label): int(np.count_nonzero(test_labels == label))
                for label in np.unique(test_labels)
            },
            "train_questions": int(len(train_group_ids)),
            "test_questions": int(len(test_group_ids)),
            "questions_in_both_splits": int(len(train_group_ids & test_group_ids)),
        },
        "model": "LogisticRegression(max_iter=1000, class_weight='balanced')",
        "metrics": {
            "accuracy": float(accuracy_score(test_labels, predictions)),
            "accuracy_ci_95_wilson": wilson_interval(correct, len(test_labels)),
            "f1": float(f1_score(test_labels, predictions, pos_label=1, zero_division=0)),
            "auc": float(roc_auc_score(test_labels, probabilities)),
        },
        "feature_cache": "training_logs/cache/text_gpt2_original_sample.npz; reused only on matching fingerprint",
    }

    output_path = ROOT / "training_logs" / "question_split.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    main()
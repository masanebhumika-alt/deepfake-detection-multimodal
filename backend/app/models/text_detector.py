import os
import re
import string

import numpy as np
import torch
import joblib
from scipy.sparse import csr_matrix, hstack
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # app/models
_ARTIFACT_DIR = os.path.join(_THIS_DIR, "artifacts", "text_transformer")
_JOBLIB_PATH = os.path.join(_THIS_DIR, "artifacts", "text_model.joblib")

_DEVICE = 0 if torch.cuda.is_available() else -1

_joblib_artifact = None
_joblib_ai_index = None
_model = None
_tokenizer = None
_ai_index = None
_zero_shot_pipe = None
_scorer = None

if os.path.isfile(_JOBLIB_PATH):
    try:
        _joblib_artifact = joblib.load(_JOBLIB_PATH)
        _classifier = _joblib_artifact["classifier"]
        _joblib_ai_index = next(
            (idx for idx, label in enumerate(_classifier.classes_) if label == 1), None
        )
        if _joblib_ai_index is None:
            raise ValueError(f"no AI class (label 1) found in classifier classes: {_classifier.classes_}")
        _scorer = "trained_joblib"
        print(f"[text_detector] loaded trained joblib model, AI class=1, classes={_classifier.classes_}")
    except Exception as e:
        print(f"[text_detector] failed to load joblib artifact, trying other scorers: {e}")
        _joblib_artifact = None

if _joblib_artifact is None and os.path.isdir(_ARTIFACT_DIR) and os.path.exists(os.path.join(_ARTIFACT_DIR, "config.json")):
    try:
        _tokenizer = AutoTokenizer.from_pretrained(_ARTIFACT_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(_ARTIFACT_DIR)
        _model.eval()
        if _DEVICE == 0:
            _model.to("cuda")

        # explicit index resolution — never assume index 1 == AI
        id2label = {int(k): v for k, v in _model.config.id2label.items()}
        _ai_index = next(
            (idx for idx, name in id2label.items() if "ai" in name.lower()), None
        )
        if _ai_index is None:
            raise ValueError(f"no 'AI' label found in id2label: {id2label}")

        _scorer = "trained_transformer"
        print(f"[text_detector] loaded fine-tuned model, AI index={_ai_index}, id2label={id2label}")
    except Exception as e:
        print(f"[text_detector] failed to load fine-tuned artifact, falling back: {e}")
        _model = None

if _joblib_artifact is None and _model is None:
    # zero-shot fallback — original off-the-shelf behavior
    _zero_shot_pipe = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=_DEVICE,
    )
    _scorer = "zero_shot_fallback"
    print("[text_detector] using zero-shot fallback (no fine-tuned artifact found)")


def _empty_response(text: str, note: str) -> dict:
    return {
        "modality": "text",
        "confidence_ai_generated": 0.0,
        "label": "Real",
        "scorer": "skipped",
        "features": {
            "character_count": len(text),
            "word_count": len(text.split()),
            "embedding_architecture": None,
        },
        "note": note,
    }


def _predict_trained(text: str) -> float:
    inputs = _tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    if _DEVICE == 0:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return float(probs[_ai_index].item())


def _predict_joblib(text: str) -> float:
    artifact = _joblib_artifact
    features = extract_features(text)
    custom_values = [[features[name] for name in artifact["custom_feature_names"]]]
    custom = csr_matrix(artifact["custom_scaler"].transform(custom_values))
    word = artifact["word_tfidf"].transform([text])
    char = artifact["char_tfidf"].transform([text])
    vector = hstack([custom, word, char])
    probabilities = artifact["classifier"].predict_proba(vector)[0]
    return float(probabilities[_joblib_ai_index])


def _predict_zero_shot(text: str) -> float:
    result = _zero_shot_pipe(text, candidate_labels=["Human", "AI-Generated"])
    labels = result["labels"]
    scores = result["scores"]
    ai_idx = labels.index("AI-Generated")
    return float(scores[ai_idx])


def predict(text: str) -> dict:
    if text is None or not text.strip():
        return _empty_response(text or "", "empty_or_whitespace_input")

    stripped = text.strip()
    char_count = len(stripped)
    word_count = len(stripped.split())

    if char_count < 3:
        resp = _empty_response(text, "micro_text_low_confidence")
        resp["features"]["character_count"] = char_count
        resp["features"]["word_count"] = word_count
        return resp

    try:
        if _joblib_artifact is not None:
            ai_prob = _predict_joblib(stripped)
            arch = "hybrid_tfidf_stylometric_classifier"
        elif _model is not None:
            ai_prob = _predict_trained(stripped)
            arch = _model.config.name_or_path or "fine_tuned_transformer"
        else:
            ai_prob = _predict_zero_shot(stripped)
            arch = "facebook/bart-large-mnli"
    except Exception as e:
        resp = _empty_response(text, f"inference_error: {e}")
        resp["scorer"] = "error_fallback"
        return resp

    confidence_ai_generated = round(ai_prob * 100, 2)
    label = "AI-Generated" if ai_prob >= 0.5 else "Real"

    return {
        "modality": "text",
        "confidence_ai_generated": confidence_ai_generated,
        "label": label,
        "scorer": _scorer,
        "features": {
            "character_count": char_count,
            "word_count": word_count,
            "embedding_architecture": arch,
        },
    }


# --- legacy handcrafted-feature extraction (used only by app.training.train_text) ---
# GPT-2 is loaded lazily on first call, not at module import, so it never affects
# the main predict() path or FastAPI startup memory/latency.
_ppl_tokenizer = None
_ppl_model = None


def _load_ppl_model():
    global _ppl_tokenizer, _ppl_model
    if _ppl_model is None:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _ppl_model = GPT2LMHeadModel.from_pretrained("gpt2")
        _ppl_model.eval()
    return _ppl_tokenizer, _ppl_model


def _compute_perplexity(text: str, max_length: int = 512) -> float:
    tokenizer, model = _load_ppl_model()
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings["input_ids"]
    if input_ids.size(1) < 2:
        return 0.0
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return float(torch.exp(outputs.loss).item())


def extract_features(text: str) -> dict:
    """Returns the 6 handcrafted stylometric features expected by
    app.training.train_text.CUSTOM_FEATURE_NAMES."""
    text = text or ""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)

    sentence_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences] or [0]
    avg_sentence_len = float(np.mean(sentence_lengths))
    sentence_len_burstiness = float(np.std(sentence_lengths))

    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0

    unique_words = {w.lower() for w in words}
    type_token_ratio = float(len(unique_words) / word_count) if word_count > 0 else 0.0

    punct_chars = [c for c in text if c in string.punctuation]
    punctuation_diversity = float(len(set(punct_chars)) / len(punct_chars)) if punct_chars else 0.0

    local_perplexity = _compute_perplexity(text)

    return {
        "avg_sentence_len": avg_sentence_len,
        "sentence_len_burstiness": sentence_len_burstiness,
        "avg_word_len": avg_word_len,
        "type_token_ratio": type_token_ratio,
        "punctuation_diversity": punctuation_diversity,
        "local_perplexity": local_perplexity,
    }
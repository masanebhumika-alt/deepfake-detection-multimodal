"""
Image detector — frequency-domain and noise-residual forensic features.

No pretrained deep model is required. It implements the classical
"frequency-artifact" forensic approach discussed in Experiment 2's
literature review (GAN images show unnatural high-frequency patterns;
diffusion images tend to be unusually smooth in noise-residual terms):

  - High-frequency energy ratio (2D FFT): fraction of spectral energy in
    the outer (high-frequency) ring of the magnitude spectrum.
  - Noise residual variance: variance of (image - denoised image); real
    camera photos carry natural sensor noise that generators often fail
    to reproduce faithfully.
  - Local Binary Pattern-style texture uniformity: a simple proxy for
    unnaturally smooth or unnaturally regular local texture.

As with the text detector, train_image.py shows how to fit a proper
classifier on these features once a real labeled dataset (e.g. the
140k Real and Fake Faces dataset) is available.
"""
import os
import numpy as np
from PIL import Image, ImageFilter

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "image_model.joblib")
_RESNET18_MODEL = None
_RESNET18_TRANSFORM = None


def load_resnet18_feature_extractor():
    global _RESNET18_MODEL, _RESNET18_TRANSFORM
    if _RESNET18_MODEL is None:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1
        _RESNET18_MODEL = resnet18(weights=weights)
        _RESNET18_MODEL.fc = torch.nn.Identity()
        _RESNET18_MODEL.eval()
        _RESNET18_MODEL.to("cpu")
        _RESNET18_TRANSFORM = weights.transforms()
    return _RESNET18_MODEL, _RESNET18_TRANSFORM


def extract_resnet18_embedding(image_path_or_file) -> np.ndarray:
    import torch

    model, transform = load_resnet18_feature_extractor()
    with Image.open(image_path_or_file) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        return model(tensor).cpu().numpy()[0]


def _load_gray(image_path_or_file, size=256) -> np.ndarray:
    img = Image.open(image_path_or_file).convert("L").resize((size, size))
    return np.asarray(img, dtype=np.float64)


def extract_features(image_path_or_file) -> dict:
    gray = _load_gray(image_path_or_file)

    # --- High-frequency energy ratio via 2D FFT ---
    f = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(f)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2)
    high_freq_mask = dist > (0.6 * max_dist)
    total_energy = magnitude.sum() + 1e-9
    high_freq_ratio = float(magnitude[high_freq_mask].sum() / total_energy)

    # --- Noise residual variance (image minus a blurred version of itself) ---
    img_pil = Image.fromarray(gray.astype(np.uint8))
    blurred = np.asarray(img_pil.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float64)
    residual = gray - blurred
    noise_residual_var = float(np.var(residual))

    # --- Local texture uniformity (std of local 3x3 variance map) ---
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(gray, 1, mode="edge")
    windows = sliding_window_view(padded, (3, 3))
    local_var = windows.reshape(*windows.shape[:2], 9).var(axis=2)
    texture_uniformity = float(1.0 / (np.std(local_var) + 1e-6))  # higher = more uniform texture

    return {
        "high_freq_energy_ratio": round(high_freq_ratio, 5),
        "noise_residual_variance": round(noise_residual_var, 4),
        "texture_uniformity": round(min(texture_uniformity, 9999.0), 4),
    }


def heuristic_score(features: dict) -> float:
    """Deterministic fallback scorer. Direction of effects follows the
    forensic-artifact literature reviewed in Experiment 2 (Face2Parts,
    Diffusion Deepfake): synthetic images often show LOW noise-residual
    variance (too clean) and HIGH texture uniformity (over-smoothed),
    with an atypical high-frequency energy profile vs. natural photos."""
    hf = features["high_freq_energy_ratio"]
    noise = features["noise_residual_variance"]
    texture = features["texture_uniformity"]

    hf_score = 1.0 - min(abs(hf - 0.02) / 0.05, 1.0)          # too little or too much is suspicious
    noise_score = max(0.0, 1.0 - min(noise / 15.0, 1.0))       # low noise -> more AI-like
    texture_score = min(texture / 50.0, 1.0)                   # very uniform texture -> more AI-like

    combined = 0.3 * hf_score + 0.4 * noise_score + 0.3 * texture_score
    return round(float(np.clip(combined * 100, 1, 99)), 2)


def predict(image_path_or_file) -> dict:
    if os.path.exists(_MODEL_PATH):
        import joblib
        clf = joblib.load(_MODEL_PATH)
        if isinstance(clf, dict) and clf.get("feature_extractor") == "resnet18_imagenet1k_v1":
            classifier = clf["classifier"]
            embedding = extract_resnet18_embedding(image_path_or_file).reshape(1, -1)
            probabilities = classifier.predict_proba(embedding)[0]
            ai_index = list(classifier.classes_).index(1)
            confidence = round(float(probabilities[ai_index] * 100), 2)
            return {
                "modality": "image",
                "confidence_ai_generated": confidence,
                "label": "Fake" if confidence >= 50 else "Real",
                "features": {
                    "embedding_architecture": "resnet18_imagenet1k_v1",
                    "embedding_dimensions": int(embedding.shape[1]),
                },
                "scorer": "trained_resnet18_embeddings",
            }

        features = extract_features(image_path_or_file)
        vec = np.array([[features["high_freq_energy_ratio"], features["noise_residual_variance"],
                          features["texture_uniformity"]]])
        probabilities = clf.predict_proba(vec)[0]
        ai_index = list(clf.classes_).index(1)
        proba = probabilities[ai_index]
        confidence = round(float(proba * 100), 2)
        source = "trained_model"
    else:
        features = extract_features(image_path_or_file)
        confidence = heuristic_score(features)
        source = "heuristic_fallback"

    return {
        "modality": "image",
        "confidence_ai_generated": confidence,
        "label": "Fake" if confidence >= 50 else "Real",
        "features": features,
        "scorer": source,
    }

"""
Confidence-calibrated fusion layer — matches the "Confidence-Calibrated
Fusion Layer" block in Experiment 4's proposed system architecture.

Rather than a hard vote, each modality's confidence contributes
proportionally to its own distance from 50 (uncertain) — a modality that
is very sure (near 0 or 100) is weighted more heavily than one sitting
near the uncertain midpoint.
"""
from typing import List, Dict


def fuse(results: List[Dict]) -> Dict:
    if not results:
        return {"fused_confidence": None, "label": "Unknown", "detail": "No modality results provided."}

    weights = []
    scores = []
    for r in results:
        conf = r["confidence_ai_generated"]
        certainty = abs(conf - 50) / 50.0  # 0 = totally uncertain, 1 = maximally certain
        weight = 0.15 + 0.85 * certainty    # floor weight so no modality is fully ignored
        weights.append(weight)
        scores.append(conf)

    total_weight = sum(weights)
    fused = sum(w * s for w, s in zip(weights, scores)) / total_weight

    if fused >= 65:
        label, tier = "Likely Fake", "High Confidence"
    elif fused >= 50:
        label, tier = "Possibly Fake", "Low Confidence"
    elif fused >= 35:
        label, tier = "Possibly Real", "Low Confidence"
    else:
        label, tier = "Likely Real", "High Confidence"

    return {
        "fused_confidence_ai_generated": round(fused, 2),
        "label": label,
        "confidence_tier": tier,
        "per_modality": results,
        "modality_weights": {r["modality"]: round(w / total_weight, 3) for r, w in zip(results, weights)},
    }

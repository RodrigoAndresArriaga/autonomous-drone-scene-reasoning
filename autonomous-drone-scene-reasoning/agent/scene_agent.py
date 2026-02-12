# Scene evaluation agent orchestration layer.

# Output contract: canonical structure for scene evaluation agent.
"""
{
    "hazards": List[{"type": str, "severity": str}],
    "drone_path_safety": {"total_risk_score": int, "classification": str},
    "human_follow_safety": {"total_risk_score": int, "classification": str},
    "recommendation": str,
    "explanation": str | None,
    "perception_complexity_score": int,
    "latency_ms": float
}
"""

import logging
import time
from pathlib import Path

from reasoning.cosmos_reasoner import query_cosmos_structured
from reasoning.explanation import generate_explanation
from reasoning.hazard_schema import HAZARD_TYPES
from safety.affordance_model import (
    DRONE_CAPABILITIES,
    HUMAN_CAPABILITIES,
    SEVERITY_WEIGHTS,
    classify_shared_safety,
)
from safety.recommendation import generate_navigation_recommendation

_last_state_signature = None
_last_explanation = None
_last_raw_extraction = None
_safe_state_memory = []
MAX_MEMORY = 5  # small rolling buffer


# Run full scene evaluation pipeline: Cosmos perception -> deterministic safety -> Cosmos explanation.
# For video: set run_hazard_extraction=False on non-key frames (e.g. frame_index % 10 != 0) to skip
# Cosmos vision encoding; deterministic layer reuses cached hazards. Use evaluate_scene(frame, run_hazard_extraction=(i % 10 == 0)).
def evaluate_scene(
    image_path: str,
    explain: bool = False,
    run_hazard_extraction: bool = True,
) -> dict:
    start = time.time()
    global _last_raw_extraction

    # 1) Cosmos structured hazard extraction (or reuse cached)
    if run_hazard_extraction or _last_raw_extraction is None:
        raw = query_cosmos_structured(image_path)
        _last_raw_extraction = raw
    else:
        raw = _last_raw_extraction

    # 2) Filter hazards and fail loudly on unknown types: filter out hazards that are not in the hazard schema.
    raw_hazards = raw.get("hazards", [])
    validated_hazards = [h for h in raw_hazards if h.get("type") in HAZARD_TYPES]
    unknown_hazards = [h for h in raw_hazards if h.get("type") not in HAZARD_TYPES]
    if unknown_hazards:
        logging.warning("Dropped unknown hazard types from Cosmos output: %s", unknown_hazards)

    # Deduplicate by hazard type, keep highest severity
    dedup = {}
    for h in validated_hazards:
        t = h["type"]
        if t not in dedup:
            dedup[t] = h
        else:
            w = SEVERITY_WEIGHTS.get(h.get("severity", "medium"), 2)
            dw = SEVERITY_WEIGHTS.get(dedup[t].get("severity", "medium"), 2)
            if w > dw:
                dedup[t] = h
    validated_hazards = list(dedup.values())

    # 3) Visibility status: get the visibility status from the Cosmos output.
    visibility_status = raw.get("visibility_status", "clear")

    # 4) Deterministic safety classification: classify the safety of the drone and human based on the hazards.
    safety = classify_shared_safety(
        validated_hazards,
        DRONE_CAPABILITIES,
        HUMAN_CAPABILITIES,
    )

    drone_class = safety["drone_path_safety"]["classification"]
    human_class = safety["human_follow_safety"]["classification"]

    global _safe_state_memory
    if drone_class == "safe" and human_class == "safe":
        # Only append when state changes (avoids redundant entries from consecutive safe frames)
        if not _safe_state_memory or _safe_state_memory[-1]["hazards"] != validated_hazards:
            _safe_state_memory.append({
                "hazards": validated_hazards,
                "drone_class": drone_class,
                "human_class": human_class,
            })
            if len(_safe_state_memory) > MAX_MEMORY:
                _safe_state_memory.pop(0)

    fallback_available = False
    if drone_class == "safe" and human_class == "unsafe" and len(_safe_state_memory) > 0:
        fallback_available = True

    # 5) Deterministic policy recommendation
    rec = generate_navigation_recommendation(
        safety["drone_path_safety"]["classification"],
        safety["human_follow_safety"]["classification"],
        visibility_status,
    )
    if fallback_available and rec["recommendation"] == "Proceed but do not guide":
        rec["recommendation"] = "Reroute (previously observed safe state available)"

    # 6) Cosmos explanation: only when decision state changes
    state_signature = (
        tuple(sorted((h["type"], h.get("severity", "medium")) for h in validated_hazards)),
        safety["drone_path_safety"]["classification"],
        safety["human_follow_safety"]["classification"],
        rec["recommendation"],
    )

    global _last_state_signature, _last_explanation
    explanation = None

    if explain:
        if state_signature != _last_state_signature:
            explanation = generate_explanation(validated_hazards, safety, rec, fallback_available)
            _last_explanation = explanation
            _last_state_signature = state_signature
        else:
            explanation = _last_explanation

    # 7) Latency: calculate the latency of the pipeline.
    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "hazards": validated_hazards,
        "drone_path_safety": safety["drone_path_safety"],
        "human_follow_safety": safety["human_follow_safety"],
        "recommendation": rec["recommendation"],
        "explanation": explanation,
        "perception_complexity_score": len(validated_hazards),
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    demo_image = project_root / "scripts" / "test_image.png"
    output = evaluate_scene(str(demo_image), explain=True)
    print(output)

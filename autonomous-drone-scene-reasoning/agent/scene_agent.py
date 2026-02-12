# Scene evaluation agent orchestration layer.

# Output contract: canonical structure for scene evaluation agent.
"""
{
    "hazards": List[{"type": str, "severity": str}],
    "drone_path_safety": {"total_risk_score": int, "classification": str},
    "human_follow_safety": {"total_risk_score": int, "classification": str},
    "recommendation": str,
    "explanation": str | None,
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
    classify_shared_safety,
)
from safety.recommendation import generate_navigation_recommendation

# Run full scene evaluation pipeline: Cosmos perception -> deterministic safety -> Cosmos explanation.
def evaluate_scene(image_path: str, explain: bool = False) -> dict:
    start = time.time()

    # 1) Cosmos structured hazard extraction: extract hazards from the image.
    raw = query_cosmos_structured(image_path)

    # 2) Filter hazards and fail loudly on unknown types: filter out hazards that are not in the hazard schema.
    raw_hazards = raw.get("hazards", [])
    validated_hazards = [h for h in raw_hazards if h.get("type") in HAZARD_TYPES]
    unknown_hazards = [h for h in raw_hazards if h.get("type") not in HAZARD_TYPES]
    if unknown_hazards:
        logging.warning("Dropped unknown hazard types from Cosmos output: %s", unknown_hazards)

    # 3) Visibility status: get the visibility status from the Cosmos output.
    visibility_status = raw.get("visibility_status", "clear")

    # 4) Deterministic safety classification: classify the safety of the drone and human based on the hazards.
    safety = classify_shared_safety(
        validated_hazards,
        DRONE_CAPABILITIES,
        HUMAN_CAPABILITIES,
    )

    # 5) Deterministic policy recommendation
    rec = generate_navigation_recommendation(
        safety["drone_path_safety"]["classification"],
        safety["human_follow_safety"]["classification"],
        visibility_status,
    )

    # 6) Optional Cosmos explanation: generate an explanation for the decision using Cosmos.
    explanation = None
    if explain:
        explanation = generate_explanation(validated_hazards, safety, rec)

    # 7) Latency: calculate the latency of the pipeline.
    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "hazards": validated_hazards, # Return the validated hazards.
        "drone_path_safety": safety["drone_path_safety"], # Return the safety classification for the drone.
        "human_follow_safety": safety["human_follow_safety"], # Return the safety classification for the human.
        "recommendation": rec["recommendation"], # Return the recommendation for the decision.
        "explanation": explanation, # Return the explanation for the decision.
        "latency_ms": latency_ms, # Return the latency of the pipeline.
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    demo_image = project_root / "scripts" / "test_image.png"
    output = evaluate_scene(str(demo_image), explain=True)
    print(output)

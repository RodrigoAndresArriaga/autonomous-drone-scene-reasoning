# Strategic explanation layer (Layer 3).
# Cosmos receives hazards, safety, and recommendation and explains the deterministic outcome.
# Does not compute it.
#
# recommendation: dict from safety.recommendation.generate_navigation_recommendation,
# possibly with fallback override from scene_agent when previously observed safe state exists.

from .cosmos_reasoner import query_cosmos_explanation


def generate_explanation(
    hazards: list[dict],
    safety: dict,
    recommendation: dict,  # from safety.recommendation.generate_navigation_recommendation
    fallback_available: bool = False,
    raw_extraction: str | None = None,
    scene_summary: str | None = None,
) -> str:
    context = {
        "hazards": hazards,
        "safety": safety,
        "recommendation": recommendation,
        "fallback_available": fallback_available,
        "raw_extraction": raw_extraction,
        "scene_summary": scene_summary or "",
    }
    return query_cosmos_explanation(context)

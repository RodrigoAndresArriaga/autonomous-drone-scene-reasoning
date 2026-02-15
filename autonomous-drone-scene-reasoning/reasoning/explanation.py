#Strategic explanation layer (Layer 3).
# Cosmos receives hazards, safety, and recommendation and explains the deterministic outcome.
# Does not compute it.

from .cosmos_reasoner import query_cosmos_explanation


def generate_explanation(
    hazards: list[dict],
    safety: dict,
    recommendation: dict,
    fallback_available: bool = False,
    raw_extraction: str | None = None,
) -> str:
    context = {
        "hazards": hazards,
        "safety": safety,
        "recommendation": recommendation,
        "fallback_available": fallback_available,
        "raw_extraction": raw_extraction,
    }
    return query_cosmos_explanation(context)

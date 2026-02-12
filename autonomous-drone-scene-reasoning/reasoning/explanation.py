#Strategic explanation layer (Layer 3).
# Cosmos receives hazards, safety, and recommendation and explains the deterministic outcome.
# Does not compute it.

from .cosmos_reasoner import query_cosmos_explanation


def generate_explanation(
    hazards: list[dict],
    safety: dict,
    recommendation: dict,
) -> str:
    """Build context and call Cosmos for strategic explanation."""
    context = {
        "hazards": hazards,
        "safety": safety,
        "recommendation": recommendation,
    }
    return query_cosmos_explanation(context)

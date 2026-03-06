# Visibility / occlusion status for forward scout logic.
# Canonical visibility statuses. Used by cosmos_reasoner (Layer 2) and scene_agent for validation.
# Feeds recommendation.generate_navigation_recommendation(visibility_status=...).

VISIBILITY_STATUSES = ("clear", "occluded", "unknown")

# Canonical hazard taxonomy for structured hazard parsing.
# Used by Layer 2 to convert free-form model output into canonical hazard objects.

from typing import TypedDict

# Canonical hazard types with default severity (constraint mapping in affordance_model)
HAZARD_TYPES = {
    "hole": {"severity": "high"},
    "unstable_ground": {"severity": "medium"},
    "narrow_passage": {"severity": "contextual"},
    "debris": {"severity": "medium"},
    "low_visibility_dropoff": {"severity": "high"},
    "partial_floor_collapse": {"severity": "high"},
    "electrical_exposure": {"severity": "critical"},
    "slippery_surface": {"severity": "medium"},
    "overhead_instability": {"severity": "high"},
    "confined_air_pocket": {"severity": "critical"},
    "heat_source_proximity": {"severity": "high"},
    "entanglement_risk": {"severity": "medium"},
    "visual_misleading_path": {"severity": "contextual"},
    "water_depth_uncertain": {"severity": "high"},
    "unstable_debris_stack": {"severity": "high"},
    "restricted_escape_route": {"severity": "contextual"},
    "blind_corner": {"severity": "contextual"},
    "unstable_vehicle": {"severity": "high"},
    "sharp_protrusion": {"severity": "medium"},
}


# Hazard instance aligned with output contract hazards array
class HazardInstance(TypedDict, total=False):
    type: str
    location: str | None
    severity: str
    zone: str  # ground | mid | overhead | unknown


# Lookup hazard type info by canonical key
def get_hazard_info(type_key: str) -> dict | None:
    return HAZARD_TYPES.get(type_key)

#Canonical hazard taxonomy for structured hazard parsing.

#Used by the Structured Hazard Parsing step to convert free-form model output into hazard objects that feed the output contract (hazards array).


from typing import TypedDict

# Canonical hazard types: affects = agent(s) impacted, severity = default severity
HAZARD_TYPES = {
    "hole": {"affects": ["human", "drone_low_altitude"], "severity": "high"},
    "unstable_ground": {"affects": ["human"], "severity": "medium"},
    "narrow_passage": {"affects": ["human", "large_drone"], "severity": "contextual"},
    "debris": {"affects": ["human"], "severity": "medium"},
    "low_visibility_dropoff": {"affects": ["human", "drone_low_altitude"],"severity": "high",},
    "partial_floor_collapse": {"affects": ["human"], "severity": "high"},
    "electrical_exposure": {"affects": ["human"], "severity": "critical"},
    "slippery_surface": {"affects": ["human"], "severity": "medium"},
    "overhead_instability": {"affects": ["human", "drone"], "severity": "high"},
    "confined_air_pocket": {"affects": ["human"], "severity": "critical"},
    "heat_source_proximity": {"affects": ["human", "drone"], "severity": "high"},
    "entanglement_risk": {"affects": ["human", "drone"], "severity": "medium"},
    "visual_misleading_path": {"affects": ["human"], "severity": "contextual"},
    "water_depth_uncertain": {"affects": ["human", "ground_robot"], "severity": "high"},
    "unstable_debris_stack": {"affects": ["human"], "severity": "high"},
    "restricted_escape_route": {"affects": ["human"], "severity": "contextual"},
    "blind_corner": {"affects": ["human", "drone"], "severity": "contextual"},
    "unstable_vehicle": {"affects": ["human"], "severity": "high"},
    "sharp_protrusion": {"affects": ["human", "drone_low_altitude"],"severity": "medium",},
}


class HazardInstance(TypedDict, total=False):
    """Hazard instance aligned with output contract hazards array."""

    type: str
    location: str | None
    severity: str
    affects_human: bool


def get_hazard_info(type_key: str) -> dict | None:
    """Lookup hazard type info by canonical key."""
    return HAZARD_TYPES.get(type_key)

# Affordance layer: maps structured hazards to drone vs human risk via capability constraints.
# Independent of vision. No geometry, no ML, deterministic only.
# Input: hazards from Layer 2. Output: drone_path_safety, human_follow_safety -> passed to recommendation.

# Capability Profiles

HUMAN_CAPABILITIES = {
    "requires_stable_ground": True,
    "requires_body_clearance": True,
    "exposed_to_falling_debris": True,
    "sensitive_to_electricity": True,
    "sensitive_to_heat": True,
    "sensitive_to_entanglement": True,
    "requires_oxygen": True,
}

DRONE_CAPABILITIES = {
    "can_fly_over_gaps": True,
    "requires_stable_ground": False,
    "requires_clear_airspace": True,
    "exposed_to_falling_debris": True,
    "sensitive_to_electricity": True,
    "sensitive_to_heat": True,
    "sensitive_to_entanglement": True,
    "requires_oxygen": False,
}

# Hazard Impact Rules

HAZARD_CONSTRAINT_MAP = {
    "hole": ["requires_stable_ground"],
    "low_visibility_dropoff": ["requires_stable_ground"],
    "unstable_ground": ["requires_stable_ground"],
    "partial_floor_collapse": ["requires_stable_ground"],
    "unstable_debris_stack": ["requires_stable_ground"],
    "slippery_surface": ["requires_stable_ground"],
    "water_depth_uncertain": ["requires_stable_ground"],
    "electrical_exposure": ["sensitive_to_electricity"],
    "heat_source_proximity": ["sensitive_to_heat"],
    "entanglement_risk": ["sensitive_to_entanglement"],
    "confined_air_pocket": ["requires_oxygen"],
    "sharp_protrusion": ["requires_stable_ground", "requires_clear_airspace"],
    "overhead_instability": ["requires_clear_airspace", "exposed_to_falling_debris"],
    "narrow_passage": ["requires_clear_airspace", "requires_body_clearance"],
    "restricted_escape_route": ["limits_escape_options"],
}

SEVERITY_WEIGHTS = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
    "contextual": 2,
}
MINIMUM_SEVERITY_FLOOR_TYPES = (
    "hole",
    "low_visibility_dropoff",
    "electrical_exposure",
    "confined_air_pocket",
    "partial_floor_collapse",
)

# Table-based severity to risk weight. Floor applied for catastrophic types.
def _get_severity_weight(hazard: dict, htype: str) -> int:
    severity = hazard.get("severity", "medium")
    base = SEVERITY_WEIGHTS.get(severity, 2)
    if htype in MINIMUM_SEVERITY_FLOOR_TYPES:
        base = max(base, SEVERITY_WEIGHTS["medium"])
    return base


# Evaluate a single hazard against an agent's capabilities. Returns (risk_score, violated_constraints).
def evaluate_hazard_for_agent(hazard: dict, capabilities: dict, agent: str) -> tuple[int, list[str]]:
    htype = hazard.get("type", "")
    zone = hazard.get("zone", "unknown")
    weight = _get_severity_weight(hazard, htype)
    risk = 0
    violated: list[str] = []

    for constraint in HAZARD_CONSTRAINT_MAP.get(htype, []):
        if constraint == "requires_stable_ground":
            if capabilities.get("requires_stable_ground") and zone in ("ground", "unknown"):
                risk += weight
                violated.append(f"{htype}: requires_stable_ground (zone={zone})")
        elif constraint == "requires_clear_airspace":
            if capabilities.get("requires_clear_airspace") and zone in ("mid", "overhead", "unknown"):
                risk += weight
                violated.append(f"{htype}: requires_clear_airspace (zone={zone})")
        elif constraint == "sensitive_to_entanglement":
            if capabilities.get("sensitive_to_entanglement"):
                if agent == "human":
                    if zone in ("ground", "mid", "overhead", "unknown"):
                        risk += weight
                        violated.append(f"{htype}: sensitive_to_entanglement (zone={zone})")
                elif agent == "drone":
                    if zone in ("mid", "overhead", "unknown"):
                        risk += weight
                        violated.append(f"{htype}: sensitive_to_entanglement (zone={zone})")
        elif constraint == "limits_escape_options":
            if agent == "human":
                risk += SEVERITY_WEIGHTS["contextual"]
                violated.append(f"{htype}: limits_escape_options")
        elif constraint == "requires_body_clearance":
            if capabilities.get("requires_body_clearance") and zone in ("mid", "overhead", "unknown"):
                risk += weight
                violated.append(f"{htype}: requires_body_clearance (zone={zone})")
        elif constraint == "exposed_to_falling_debris":
            if capabilities.get("exposed_to_falling_debris") and zone in ("overhead", "unknown"):
                risk += weight
                violated.append(f"{htype}: exposed_to_falling_debris (zone={zone})")
        elif capabilities.get(constraint, False):
            risk += weight
            violated.append(f"{htype}: {constraint}")

    return risk, violated

# Map total risk score to safety classification.
def _risk_to_classification(score: int) -> str:
    if score <= 3:
        return "safe"
    if score <= 7:
        return "caution"
    return "unsafe"


# Path-Level Classification and Risk Interpretation

def classify_shared_safety(
    hazards: list[dict],
    drone_capabilities: dict,
    human_capabilities: dict,
) -> dict:
    """Classify shared path safety for drone and human based on hazards and capabilities."""
    drone_risk = 0
    human_risk = 0
    drone_violated: list[str] = []
    human_violated: list[str] = []
    violations: list[dict] = []

    for h in hazards:
        dr, dv = evaluate_hazard_for_agent(h, drone_capabilities, "drone")
        hr, hv = evaluate_hazard_for_agent(h, human_capabilities, "human")
        drone_risk += dr
        human_risk += hr
        drone_violated.extend(dv)
        human_violated.extend(hv)
        combined = list(dict.fromkeys(dv + hv))
        if combined:
            violations.append({
                "type": h.get("type", ""),
                "zone": h.get("zone", "unknown"),
                "violated": combined,
            })

    return {
        "drone_path_safety": {
            "total_risk_score": drone_risk,
            "classification": _risk_to_classification(drone_risk),
            "violated_constraints": drone_violated,
        },
        "human_follow_safety": {
            "total_risk_score": human_risk,
            "classification": _risk_to_classification(human_risk),
            "violated_constraints": human_violated,
        },
        "violations": violations,
    }


# Minimal Inline Test to test the affordance model

if __name__ == "__main__":
    hazards = [
        {"type": "hole", "severity": "high", "zone": "ground"},
        {"type": "electrical_exposure", "severity": "critical", "zone": "mid"},
        {"type": "unstable_ground", "severity": "medium", "zone": "ground"},
    ]
    result = classify_shared_safety(hazards, DRONE_CAPABILITIES, HUMAN_CAPABILITIES)
    print(result)

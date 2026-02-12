#Affordance layer: maps structured hazards to drone vs human risk via capability constraints. Independent of vision. No geometry, no ML, deterministic only.

# Capability Profiles

DRONE_CAPABILITIES = {
    "can_fly_over_gaps": True,
    "requires_stable_ground": False,
    "sensitive_to_electricity": True,
    "sensitive_to_heat": True,
    "sensitive_to_entanglement": True,
    "requires_oxygen": False,
}

HUMAN_CAPABILITIES = {
    "can_fly_over_gaps": False,
    "requires_stable_ground": True,
    "sensitive_to_electricity": True,
    "sensitive_to_heat": True,
    "sensitive_to_entanglement": True,
    "requires_oxygen": True,
}

# Hazard Impact Rules

GAP_HAZARDS = ("hole", "low_visibility_dropoff")
GROUND_INSTABILITY_HAZARDS = ("unstable_ground", "partial_floor_collapse", "unstable_debris_stack")

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
def evaluate_hazard_for_agent(hazard: dict, capabilities: dict) -> tuple[int, list[str]]:
    htype = hazard.get("type", "")
    risk = 0
    violated: list[str] = []

    # GAP hazards: agent cannot fly over
    if htype in GAP_HAZARDS and not capabilities.get("can_fly_over_gaps", False):
        risk += _get_severity_weight(hazard, htype)
        violated.append(f"{htype}: cannot fly over gaps")

    # Ground instability: agent requires stable ground
    if htype in GROUND_INSTABILITY_HAZARDS and capabilities.get("requires_stable_ground", False):
        risk += _get_severity_weight(hazard, htype)
        violated.append(f"{htype}: requires stable ground")

    # Electrical exposure: agent is sensitive to electricity
    if htype == "electrical_exposure" and capabilities.get("sensitive_to_electricity", False):
        risk += _get_severity_weight(hazard, htype)
        violated.append("electrical_exposure: sensitive to electricity")

    # Heat source proximity: agent is sensitive to heat
    if htype == "heat_source_proximity" and capabilities.get("sensitive_to_heat", False):
        risk += _get_severity_weight(hazard, htype)
        violated.append("heat_source_proximity: sensitive to heat")

    # Entanglement risk: agent is sensitive to entanglement
    if htype == "entanglement_risk" and capabilities.get("sensitive_to_entanglement", False):
        risk += _get_severity_weight(hazard, htype)
        violated.append("entanglement_risk: sensitive to entanglement")

    # Confined air pocket: agent requires oxygen
    if htype == "confined_air_pocket" and capabilities.get("requires_oxygen", False):
        risk += _get_severity_weight(hazard, htype)
        violated.append("confined_air_pocket: requires oxygen")

    return risk, violated

# Map total risk score to safety classification.
def _risk_to_classification(score: int) -> str:
    if score <= 4:
        return "safe"
    if score <= 9:
        return "caution"
    return "unsafe"


# Path-Level Classification and Risk Interpretation

def classify_shared_safety(
    hazards: list[dict],
    drone_capabilities: dict,
    human_capabilities: dict) -> dict:
    # Classify shared path safety for drone and human based on hazards and capabilities.
    drone_risk = 0
    human_risk = 0

    # Evaluate hazards for drone and human
    for h in hazards:
        dr, _ = evaluate_hazard_for_agent(h, drone_capabilities)
        hr, _ = evaluate_hazard_for_agent(h, human_capabilities)
        drone_risk += dr
        human_risk += hr

    # Return the safety classifications for drone and human
    return {
        "drone_path_safety": {
            "total_risk_score": drone_risk,
            "classification": _risk_to_classification(drone_risk),
        },
        "human_follow_safety": {
            "total_risk_score": human_risk,
            "classification": _risk_to_classification(human_risk),
        },
    }


# Minimal Inline Test to test the affordance model

if __name__ == "__main__":
    hazards = [
        {"type": "hole", "severity": "high"},
        {"type": "electrical_exposure", "severity": "critical"},
        {"type": "unstable_ground", "severity": "medium"},
    ]
    result = classify_shared_safety(hazards, DRONE_CAPABILITIES, HUMAN_CAPABILITIES)
    print(result)

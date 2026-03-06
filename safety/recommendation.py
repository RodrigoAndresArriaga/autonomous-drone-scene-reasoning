# Navigation recommendation logic: deterministic decision table. Maps drone + human classification to a single explicit recommendation.
# No risk calculation. No external imports.
# Inputs from affordance_model (drone_class, human_class) and Layer 2 (visibility_status). Output drives Layer 3 explanation.

RECOMMENDATIONS = {
    "PROCEED_AND_GUIDE": "Proceed and guide human",
    "PROCEED_DRONE_ONLY": "Proceed but do not guide",
    "REROUTE": "Reroute before guiding",
    "HOLD": "Hold position",
    "SCOUT_FORWARD": "Scout forward before guiding human",
}

# Convert drone and human classification strings to one explicit recommendation.
def generate_navigation_recommendation(
    drone_classification: str,
    human_classification: str,
    visibility_status: str = "clear",
) -> dict:
    # Rule 0: Scout case: occluded ahead, both safe → drone scouts alone
    if visibility_status == "occluded" and drone_classification == "safe" and human_classification == "safe":
        return {
            "recommendation": RECOMMENDATIONS["SCOUT_FORWARD"],
            "drone_status": drone_classification,
            "human_status": human_classification,
        }

    # Rule 1: Drone unsafe: cannot proceed
    if drone_classification == "unsafe":
        return {
            "recommendation": RECOMMENDATIONS["REROUTE"],
            "drone_status": drone_classification,
            "human_status": human_classification,
        }

    # Rule 2: Both safe
    if drone_classification == "safe" and human_classification == "safe":
        return {
            "recommendation": RECOMMENDATIONS["PROCEED_AND_GUIDE"],
            "drone_status": drone_classification,
            "human_status": human_classification,
        }

    # Rule 3: Drone safe but human unsafe
    if drone_classification == "safe" and human_classification == "unsafe":
        return {
            "recommendation": RECOMMENDATIONS["PROCEED_DRONE_ONLY"],
            "drone_status": drone_classification,
            "human_status": human_classification,
        }

    # Rule 4: Any caution case
    if "caution" in [drone_classification, human_classification]:
        return {
            "recommendation": RECOMMENDATIONS["HOLD"],
            "drone_status": drone_classification,
            "human_status": human_classification,
        }

    # Fallback: Hold position if no other rule applies
    return {
        "recommendation": RECOMMENDATIONS["HOLD"],
        "drone_status": drone_classification,
        "human_status": human_classification,
    }


# Minimal Inline Test to test the recommendation logic
if __name__ == "__main__":
    scenarios = [
        ("safe", "safe"),
        ("safe", "unsafe"),
        ("unsafe", "safe"),
        ("caution", "safe"),
        ("safe", "safe", "occluded"),
    ]
    for scenario in scenarios:
        if len(scenario) == 3:
            drone, human, vis = scenario
            result = generate_navigation_recommendation(drone, human, visibility_status=vis)
        else:
            drone, human = scenario
            result = generate_navigation_recommendation(drone, human)
        print(result)

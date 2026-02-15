"""Format evaluation results for judge-ready console output."""


def format_results(
    rows: list[dict],
    mode: str,
    input_path: str,
    metrics: dict,
    total_latency_s: float,
) -> str:
    """Produce structured, judge-legible output from evaluation rows and metrics."""
    if not rows:
        return "No results."

    display_row = rows[-1]

    # Step 2 — Header
    lines = [
        "=== Autonomous Drone Scene Reasoning ===",
        "",
        f"Input: {input_path}",
        f"Mode: {mode}",
        "",
    ]

    # Step 3 — Scene Summary
    scene_summary = display_row.get("scene_summary") or "N/A"
    lines.extend(["Scene Summary:", scene_summary, ""])

    # Step 4 — Hazards
    hazards = display_row.get("hazards", [])
    lines.append("Detected Hazards:")
    if not hazards:
        lines.append("None")
    else:
        for h in hazards:
            t = h.get("type", "unknown")
            s = h.get("severity", "medium")
            z = h.get("zone", "unknown")
            lines.append(f"- {t} ({s}, {z})")
    lines.append("")

    # Step 5 — Safety
    drone = display_row.get("drone_path_safety", {})
    human = display_row.get("human_follow_safety", {})
    drone_class = drone.get("classification", "unknown")
    drone_risk = drone.get("total_risk_score", 0)
    human_class = human.get("classification", "unknown")
    human_risk = human.get("total_risk_score", 0)
    lines.extend([
        f"Drone Path Safety: {drone_class} (risk={drone_risk})",
        f"Human Follow Safety: {human_class} (risk={human_risk})",
        "",
    ])

    # Step 6 — Deterministic Recommendation
    rec = display_row.get("recommendation")
    rec_text = rec.get("recommendation", "N/A") if isinstance(rec, dict) else str(rec)
    lines.extend(["Deterministic Recommendation:", rec_text, ""])

    # Step 7 — Explanation
    lines.append("Explanation:")
    exp = display_row.get("explanation")
    if exp:
        lines.append(exp if isinstance(exp, str) else str(exp))
    else:
        lines.append("(disabled)")
    lines.append("")

    # Step 8 — Metrics
    frames_total = metrics.get("frames_total", 0)
    frames_with_hazards = metrics.get("frames_with_hazards", 0)
    pct = 100 * frames_with_hazards / frames_total if frames_total else 0
    norm_invocations = metrics.get("normalization_triggered", 0)
    json_failures = metrics.get("json_parse_failures", 0)

    lines.extend([
        "--- Evaluation Metrics ---",
        f"Frames Evaluated: {frames_total}",
        f"Frames with Hazards: {frames_with_hazards}",
        f"Non-Empty Hazard Rate: {pct:.0f}%",
        f"Normalization Invocations: {norm_invocations}",
        f"JSON Parse Failures: {json_failures}",
        "",
    ])

    # Step 9 — Latency
    lines.append(f"Total Latency: {total_latency_s:.1f}s")

    return "\n".join(lines)

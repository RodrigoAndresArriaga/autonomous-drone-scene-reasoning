# Scene evaluation agent orchestration layer.

# We support frame, clip, and rolling-window inference. Rolling-window emulates live
# operation by analyzing the most recent N seconds repeatedly. This adds temporal
# continuity without adding mapping/planning/control.

# Output contract: canonical structure for scene evaluation agent.
"""
{
    "hazards": List[{"type": str, "severity": str}],
    "drone_path_safety": {"total_risk_score": int, "classification": str},
    "human_follow_safety": {"total_risk_score": int, "classification": str},
    "recommendation": dict,  # full rec from generate_navigation_recommendation
    "explanation": str | None,
    "perception_complexity_score": int,
    "latency_ms": float
}
"""

import logging
import os
import time
from pathlib import Path
from typing import Literal

from reasoning.cosmos_reasoner import (
    extract_scene_summary_from_layer1,
    get_cosmos_eval_counts,
    query_cosmos_extract,
    query_cosmos_normalize,
)
from reasoning.explanation import generate_explanation
from reasoning.hazard_schema import HAZARD_TYPES
from configs.config import get_config
from safety.affordance_model import (
    DRONE_CAPABILITIES,
    HUMAN_CAPABILITIES,
    SEVERITY_WEIGHTS,
    classify_shared_safety,
)
from safety.recommendation import RECOMMENDATIONS, generate_navigation_recommendation
from safety.scouting import VISIBILITY_STATUSES
from utils.video_clip import extract_clip_ctx, get_video_duration

# Cache state and explanation to avoid redundant LLM calls
_last_state_signature = None
_last_explanation = None
_last_raw_extraction = None
_safe_state_memory = []

# Evaluation counters for judge credibility and metrics reporting
_frames_total = 0
_frames_with_hazards = 0
_normalization_triggered_count = 0


def get_evaluation_metrics() -> dict:
    """Return evaluation counters for format_results."""
    cosmos = get_cosmos_eval_counts()
    return {
        "frames_total": _frames_total,
        "frames_with_hazards": _frames_with_hazards,
        "normalization_triggered": _normalization_triggered_count,
        "json_parse_failures": cosmos["json_parse_failures"],
        "normalization_parse_failures": cosmos["normalization_parse_failures"],
    }


# Run full scene evaluation pipeline: Cosmos perception -> deterministic safety -> Cosmos explanation.
# Modes: image (single frame), video (full file or end-window), rolling (clipped windows).
def evaluate_scene(
    image_path: str | None = None,
    video_path: str | None = None,
    *,
    mode: Literal["image", "video", "rolling"] = "image",
    fps: int | None = None,
    clip_seconds: float | None = None,
    step_seconds: float | None = None,
    explain: bool = False,
    run_hazard_extraction: bool = True,
) -> dict:
    global _last_raw_extraction, _frames_total, _frames_with_hazards, _normalization_triggered_count

    cfg = get_config().agent
    if fps is None:
        fps = cfg.fps_default

    if image_path is None and video_path is None:
        raise ValueError("Provide image_path (mode='image') or video_path (mode='video'|'rolling')")
    if mode == "image":
        if image_path is None:
            raise ValueError("mode='image' requires image_path")
        media_path = image_path
        media_type = "image"
        needs_clip = False
        clip_sec = None
    elif mode == "video":
        if video_path is None:
            raise ValueError("mode='video' requires video_path")
        media_path = video_path
        media_type = "video"
        needs_clip = clip_seconds is not None and clip_seconds > 0
        clip_sec = clip_seconds if needs_clip else None
    elif mode == "rolling":
        if video_path is None:
            raise ValueError("mode='rolling' requires video_path")
        media_path = video_path
        media_type = "video"
        needs_clip = True
        clip_sec = clip_seconds if clip_seconds is not None else cfg.clip_seconds
    else:
        raise ValueError(f"mode must be 'image', 'video', or 'rolling', got {mode!r}")

    # Extract end-anchored clip: last N seconds for rolling window behavior
    raw = None
    if needs_clip and clip_sec and clip_sec > 0:
        total_duration = get_video_duration(media_path)
        start_sec = max(0.0, total_duration - clip_sec)
        with extract_clip_ctx(media_path, start_sec, clip_sec, reencode_fps=fps) as clip_path:
            actual_media_path = str(clip_path)
            # Extract happens inside this block; Cosmos call must be inside for temp to exist
            if run_hazard_extraction or _last_raw_extraction is None:
                raw = query_cosmos_extract(actual_media_path, media_type="video", fps=fps)
    else:
        actual_media_path = media_path

    start = time.time()
    _frames_total += 1

    # 1) Cosmos structured hazard extraction (or reuse cached)
    if run_hazard_extraction or _last_raw_extraction is None:
        if raw is None:
            raw = query_cosmos_extract(actual_media_path, media_type=media_type, fps=fps)
        _last_raw_extraction = raw
    else:
        raw = _last_raw_extraction

    # 2) Layer 2 interprets raw output and maps hazards to canonical types.
    _normalization_triggered_count += 1
    raw_text = raw.get("raw", "")
    norm_result = query_cosmos_normalize(raw_text)
    normalized = norm_result.get("hazards", [])
    visibility_status = norm_result.get("visibility_status", "unknown")
    visibility_status = visibility_status if visibility_status in VISIBILITY_STATUSES else "unknown"
    scene_summary = norm_result.get("scene_summary", "") or extract_scene_summary_from_layer1(raw_text)

    # Collapse duplicate hazards by (type, zone), keeping highest severity
    dedup: dict[tuple[str, str], dict] = {}
    for h in normalized:
        t = h.get("type", "")
        z = h.get("zone", "unknown")
        key = (t, z)
        if key not in dedup:
            dedup[key] = h
        else:
            w = SEVERITY_WEIGHTS.get(h.get("severity", "medium"), 2)
            dw = SEVERITY_WEIGHTS.get(dedup[key].get("severity", "medium"), 2)
            if w > dw:
                dedup[key] = h
    validated_hazards = list(dedup.values())
    if validated_hazards:
        _frames_with_hazards += 1

    # Apply deterministic affordance-based safety classification
    safety = classify_shared_safety(
        validated_hazards,
        DRONE_CAPABILITIES,
        HUMAN_CAPABILITIES,
    )

    drone_class = safety["drone_path_safety"]["classification"]
    human_class = safety["human_follow_safety"]["classification"]

    if os.environ.get("COSMOS_TIMING") == "1":
        print("[OUR computation] Step 3: Deterministic safety classification (affordance model):")
        print(f"  drone_path_safety={drone_class}, human_follow_safety={human_class}")

    # Track safe states for potential fallback rerouting
    max_memory = get_config().agent.memory.max_memory
    global _safe_state_memory
    if drone_class == "safe" and human_class == "safe":
        # Only append when state changes (avoids redundant entries from consecutive safe frames)
        if not _safe_state_memory or _safe_state_memory[-1]["hazards"] != validated_hazards:
            _safe_state_memory.append({
                "hazards": validated_hazards,
                "drone_class": drone_class,
                "human_class": human_class,
            })
            if len(_safe_state_memory) > max_memory:
                _safe_state_memory.pop(0)

    # Check if we can reroute to a previously observed safe state
    fallback_available = False
    if drone_class == "safe" and human_class == "unsafe" and len(_safe_state_memory) > 0:
        fallback_available = True

    # 4) Deterministic policy recommendation
    # recommendation: Rule 3: drone=safe, human=unsafe -> "Proceed but do not guide" (RECOMMENDATIONS["PROCEED_DRONE_ONLY"])
    rec = generate_navigation_recommendation(
        safety["drone_path_safety"]["classification"],
        safety["human_follow_safety"]["classification"],
        visibility_status,
    )
    # Override with fallback reroute when safe state memory exists
    if fallback_available and rec["recommendation"] == "Proceed but do not guide":
        rec["recommendation"] = "Reroute (previously observed safe state available)"
        if os.environ.get("COSMOS_TIMING") == "1":
            print("[OUR computation] Fallback override applied: Reroute (previously observed safe state available)")

    if os.environ.get("COSMOS_TIMING") == "1":
        print(f"[OUR computation] Available (RECOMMENDATIONS): {' | '.join(RECOMMENDATIONS.values())}")
        print(f"[OUR computation] Step 4: OUR deterministic recommendation: \"{rec['recommendation']}\"")
        print("  (from safety.recommendation.generate_navigation_recommendation - LLM does NOT decide this)")

    # Build state signature for explanation caching
    hazard_tuples = [
        (h["type"], h.get("severity", "medium"), h.get("zone", "unknown"))
        for h in validated_hazards
    ]
    state_signature = (
        tuple(sorted(hazard_tuples)),
        safety["drone_path_safety"]["classification"],
        safety["human_follow_safety"]["classification"],
        rec["recommendation"],
    )

    global _last_state_signature, _last_explanation
    explanation = None

    # Generate explanation only when state changes to reduce LLM calls
    if explain:
        if state_signature != _last_state_signature:
            if os.environ.get("COSMOS_TIMING") == "1":
                print("[OUR computation] Step 5: Passing OUR recommendation to LLM for explanation only. LLM explains our decision; it does not compute it.")
            # rec from generate_navigation_recommendation (safety.recommendation), passed to explanation layer
            explanation = generate_explanation(
                validated_hazards, safety, rec, fallback_available, raw_extraction=raw_text, scene_summary=scene_summary
            )
            _last_explanation = explanation
            _last_state_signature = state_signature
        else:
            explanation = _last_explanation

    # Calculate total pipeline latency
    latency_ms = round((time.time() - start) * 1000, 2)

    # Log evaluation metrics periodically when timing mode is enabled
    _eval_log_interval = get_config().agent.eval_log_every
    if os.environ.get("COSMOS_TIMING") == "1":
        if _frames_total == 1 or _frames_total % _eval_log_interval == 0:
            cosmos_counts = get_cosmos_eval_counts()
            msg = (
                f"Cosmos eval: frames_total={_frames_total} frames_with_hazards={_frames_with_hazards} "
                f"pct_non_empty={100.0 * _frames_with_hazards / _frames_total if _frames_total else 0:.1f}% "
                f"json_parse_failures={cosmos_counts['json_parse_failures']} "
                f"normalization_triggered={_normalization_triggered_count} "
                f"normalization_parse_failures={cosmos_counts['normalization_parse_failures']}"
            )
            logging.info("%s", msg)
            print(msg)

    return {
        "hazards": validated_hazards,
        "drone_path_safety": safety["drone_path_safety"],
        "human_follow_safety": safety["human_follow_safety"],
        "recommendation": rec,
        "explanation": explanation,
        "scene_summary": scene_summary,
        "perception_complexity_score": len(validated_hazards),
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    demo_image = project_root / "scripts" / "test_image.png"
    output = evaluate_scene(str(demo_image), explain=True)
    print(output)

"""
Run scene evaluation over scenario videos.
Modes: video (full file) or rolling (windowed clips).
Output: JSONL to outputs/scenario_rollup.jsonl.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root so agent, configs, utils import
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from agent.scene_agent import evaluate_scene
from configs.config import get_config
from utils.video_clip import extract_clip_ctx, get_video_duration


def _scenario_id(path: Path, index: int) -> str:
    return path.stem or f"scenario_{index}"


def run_video_mode(video_path: Path, explain: bool) -> dict:
    """Single evaluation on full video."""
    result = evaluate_scene(
        video_path=str(video_path),
        mode="video",
        explain=explain,
    )
    return {
        "scenario_id": video_path.stem,
        "t_start": 0.0,
        "t_end": get_video_duration(str(video_path)),
        "hazards": result["hazards"],
        "drone_safe": result["drone_path_safety"]["classification"],
        "human_safe": result["human_follow_safety"]["classification"],
        "recommendation": result["recommendation"],
        "explanation": result.get("explanation"),
        "latency_ms": result["latency_ms"],
    }


def run_rolling_mode(
    video_path: Path, scenario_id: str, clip_sec: float, step_sec: float, fps: int, explain: bool
) -> list[dict]:
    """Rolling windows: forward step t=0, step, 2*step, ..."""
    duration = get_video_duration(str(video_path))
    rows = []
    t = 0.0
    while t + clip_sec <= duration:
        t_end = t + clip_sec
        with extract_clip_ctx(str(video_path), t, clip_sec, reencode_fps=fps) as clip_path:
            result = evaluate_scene(
                video_path=str(clip_path),
                mode="video",
                fps=fps,
                explain=explain,
            )
        rows.append({
            "scenario_id": scenario_id,
            "t_start": round(t, 2),
            "t_end": round(t_end, 2),
            "hazards": result["hazards"],
            "drone_safe": result["drone_path_safety"]["classification"],
            "human_safe": result["human_follow_safety"]["classification"],
            "recommendation": result["recommendation"],
            "explanation": result.get("explanation"),
            "latency_ms": result["latency_ms"],
        })
        t += step_sec
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run scene evaluation over scenario videos")
    parser.add_argument("videos", nargs="+", type=Path, help="Video file paths")
    parser.add_argument(
        "--mode",
        choices=["video", "rolling"],
        default="rolling",
        help="video=full file, rolling=windowed clips (default)",
    )
    parser.add_argument("--clip-seconds", type=float, default=None, help="Override config clip_seconds")
    parser.add_argument("--step-seconds", type=float, default=None, help="Override config step_seconds")
    parser.add_argument("--explain", action="store_true", help="Generate explanations")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSONL path (default: outputs/scenario_rollup.jsonl)")
    args = parser.parse_args()

    cfg = get_config().agent
    clip_sec = args.clip_seconds if args.clip_seconds is not None else cfg.clip_seconds
    step_sec = args.step_seconds if args.step_seconds is not None else cfg.step_seconds
    fps = cfg.fps_default

    out_path = args.output or (_project_root / "outputs" / "scenario_rollup.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for i, video_path in enumerate(args.videos):
        video_path = video_path.resolve()
        if not video_path.exists():
            print(f"Skip (not found): {video_path}", file=sys.stderr)
            continue
        scenario_id = _scenario_id(video_path, i)
        if args.mode == "video":
            row = run_video_mode(video_path, args.explain)
            all_rows.append(row)
        else:
            rows = run_rolling_mode(
                video_path, scenario_id, clip_sec, step_sec, fps, args.explain
            )
            all_rows.extend(rows)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_rows)} rows to {out_path}")
    if args.explain and all_rows:
        for i, row in enumerate(all_rows):
            exp = row.get("explanation")
            if exp:
                if not isinstance(exp, str):
                    print(f"\n--- WARNING: explanation is {type(exp)}, expected str ---")
                elif exp.strip().startswith("{") or "'type':" in exp[:200]:
                    print("\n--- WARNING: explanation may be echoed data, not prose ---")
                header = f"\n--- Layer 3 Explanation"
                if len(all_rows) > 1:
                    header += f" (row {i + 1})"
                print(header)
                print(exp)


if __name__ == "__main__":
    main()

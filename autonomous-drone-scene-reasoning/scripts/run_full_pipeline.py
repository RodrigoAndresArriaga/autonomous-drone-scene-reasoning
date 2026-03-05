"""
Full pipeline coordinator: generate → evaluate → metrics.

Runs all five stages sequentially so VRAM is fully released between
the generation models (Cosmos Predict / Transfer) and the reasoning
model (Cosmos Reason 2). Each stage is a separate subprocess.

Stages:
  1. generate_raw.py       → data/raw/        (5 synthetic baseline clips)
  2. generate_transfer.py  → data/transfer/   (20 domain-shifted variants)
  3. generate_predict.py   → data/predict/    (15 hazard-category clips)
  4. run_scenarios.py      → outputs/scenario_rollup.jsonl
  5. compute_metrics.py    → outputs/metrics_report.json  (prints F1, confusion matrix)

Freeze guard: generation scripts skip existing files by default.
Use --force-generation to pass --force to all three generation scripts.
Use --skip-generation to jump straight to evaluation (clips already on disk).

Usage:
    # Full run (first time):
    python scripts/run_full_pipeline.py \\
        --cosmos-predict-dir  /home/ubuntu/cosmos/cosmos-predict \\
        --cosmos-transfer-dir /home/ubuntu/cosmos/cosmos-transfer

    # Re-evaluate only (clips already generated):
    python scripts/run_full_pipeline.py \\
        --skip-generation

    # Force-regenerate everything:
    python scripts/run_full_pipeline.py \\
        --cosmos-predict-dir  /home/ubuntu/cosmos/cosmos-predict \\
        --cosmos-transfer-dir /home/ubuntu/cosmos/cosmos-transfer \\
        --force-generation
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
DATA = PROJECT_ROOT / "data"
OUTPUTS = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full generation + evaluation pipeline sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cosmos-predict-dir",
        type=Path,
        default=None,
        help="Path to cosmos-predict repo. Required unless --skip-generation is set.",
    )
    parser.add_argument(
        "--cosmos-transfer-dir",
        type=Path,
        default=None,
        help="Path to cosmos-transfer repo. Required unless --skip-generation is set.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        default=False,
        help="Skip all three generation stages (clips must already be in data/).",
    )
    parser.add_argument(
        "--force-generation",
        action="store_true",
        default=False,
        help="Pass --force to generation scripts, regenerating all clips from scratch.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["video", "rolling"],
        default="video",
        help="Evaluation mode for run_scenarios.py (default: video — one pass per clip).",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        default=False,
        help="Disable Layer 3 explanations during evaluation (faster).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Duration in seconds for generated clips (default: 6.0).",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="768x512",
        help="Resolution for generated clips (default: 768x512).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="FPS for generated clips (default: 4).",
    )
    return parser.parse_args()


def _run(cmd: list, label: str) -> None:
    """Run a subprocess command, printing a banner and timing it. Exit on failure."""
    print(f"\n{'='*60}")
    print(f"  STAGE: {label}")
    print(f"  CMD  : {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(
            f"\nError: stage '{label}' failed with exit code {result.returncode}.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)
    print(f"\n  Stage '{label}' completed in {elapsed:.1f}s")


def _glob_videos(directory: Path) -> list[str]:
    """Return sorted list of .mp4 paths in directory as strings."""
    return sorted(str(p) for p in directory.glob("*.mp4"))


def main() -> None:
    args = parse_args()

    if not args.skip_generation:
        if args.cosmos_predict_dir is None:
            print(
                "Error: --cosmos-predict-dir is required unless --skip-generation is set.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.cosmos_transfer_dir is None:
            print(
                "Error: --cosmos-transfer-dir is required unless --skip-generation is set.",
                file=sys.stderr,
            )
            sys.exit(1)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # Stage 1 — Generate raw baseline clips (Cosmos Predict)
    # ------------------------------------------------------------------
    if not args.skip_generation:
        cmd = [
            sys.executable, str(SCRIPTS / "generate_raw.py"),
            "--cosmos-predict-dir", str(args.cosmos_predict_dir),
            "--duration", str(args.duration),
            "--resolution", args.resolution,
            "--fps", str(args.fps),
        ]
        if args.force_generation:
            cmd.append("--force")
        _run(cmd, "generate_raw → data/raw/ (5 clips)")

    # ------------------------------------------------------------------
    # Stage 2 — Generate Transfer variants (Cosmos Transfer)
    # ------------------------------------------------------------------
    if not args.skip_generation:
        cmd = [
            sys.executable, str(SCRIPTS / "generate_transfer.py"),
            "--cosmos-transfer-dir", str(args.cosmos_transfer_dir),
            "--resolution", args.resolution,
            "--fps", str(args.fps),
        ]
        if args.force_generation:
            cmd.append("--force")
        _run(cmd, "generate_transfer → data/transfer/ (20 clips)")

    # ------------------------------------------------------------------
    # Stage 3 — Generate Predict clips (Cosmos Predict)
    # ------------------------------------------------------------------
    if not args.skip_generation:
        cmd = [
            sys.executable, str(SCRIPTS / "generate_predict.py"),
            "--cosmos-predict-dir", str(args.cosmos_predict_dir),
            "--duration", str(args.duration),
            "--resolution", args.resolution,
            "--fps", str(args.fps),
        ]
        if args.force_generation:
            cmd.append("--force")
        _run(cmd, "generate_predict → data/predict/ (15 clips)")

    # ------------------------------------------------------------------
    # Stage 4 — Run Cosmos Reason 2 evaluation across all 40 clips
    # ------------------------------------------------------------------
    raw_clips = _glob_videos(DATA / "raw")
    transfer_clips = _glob_videos(DATA / "transfer")
    predict_clips = _glob_videos(DATA / "predict")
    all_clips = raw_clips + transfer_clips + predict_clips

    if not all_clips:
        print(
            "Error: no .mp4 files found in data/raw/, data/transfer/, or data/predict/. "
            "Run generation stages first.",
            file=sys.stderr,
        )
        sys.exit(1)

    rollup_path = OUTPUTS / "scenario_rollup.jsonl"
    cmd = [
        sys.executable, str(SCRIPTS / "run_scenarios.py"),
        "--mode", args.eval_mode,
        "--output", str(rollup_path),
    ] + all_clips

    if args.no_explain:
        cmd.append("--no-explain")

    _run(cmd, f"run_scenarios ({len(all_clips)} clips, mode={args.eval_mode})")

    # ------------------------------------------------------------------
    # Stage 5 — Compute benchmark metrics
    # ------------------------------------------------------------------
    metrics_path = OUTPUTS / "metrics_report.json"
    cmd = [
        sys.executable, str(SCRIPTS / "compute_metrics.py"),
        "--metadata", str(DATA / "metadata.csv"),
        "--rollup", str(rollup_path),
        "--output", str(metrics_path),
    ]
    _run(cmd, "compute_metrics → outputs/metrics_report.json")

    total = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE — total time: {total/60:.1f} min")
    print(f"  Rollup  : {rollup_path}")
    print(f"  Metrics : {metrics_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

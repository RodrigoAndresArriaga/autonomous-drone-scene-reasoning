"""
Generate 5 synthetic baseline clips for data/raw/ using Cosmos Predict.

Since no real R6 clips are available, this script generates the raw-tier
baseline using single-path corridor prompts defined in configs/raw_prompts.yaml.
One clip is generated per canonical hazard type (5 total). These clips are then
used as source material for Cosmos Transfer (generate_transfer.py) to produce
20 domain-shifted variants in data/transfer/.

Clips are named:  raw_{hazard_type}.mp4

Generation is skipped by default when an output file already exists.
Use --force to regenerate existing files.

After generation, writes/updates data/generation_manifest.json with the
raw provenance block.

Usage:
    python scripts/generate_raw.py \\
        --cosmos-predict-dir /home/ubuntu/cosmos/cosmos-predict

    # Regenerate everything from scratch:
    python scripts/generate_raw.py \\
        --cosmos-predict-dir /home/ubuntu/cosmos/cosmos-predict \\
        --force

    # Custom duration:
    python scripts/generate_raw.py \\
        --cosmos-predict-dir /home/ubuntu/cosmos/cosmos-predict \\
        --duration 8
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = PROJECT_ROOT / "configs" / "raw_prompts.yaml"

DATASET_VERSION = "v1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 5 synthetic raw baseline clips via Cosmos Predict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory to write raw baseline clips (default: data/raw).",
    )
    parser.add_argument(
        "--cosmos-predict-dir",
        type=Path,
        default=None,
        help=(
            "Path to the cosmos-predict repo on this machine. "
            "Falls back to COSMOS_PREDICT_DIR env var if not provided."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Duration in seconds for each generated clip (default: 6.0).",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="768x512",
        help="Target resolution for generated clips, e.g. 768x512 (default: 768x512).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second for generated clips (default: 4).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Regenerate output files even if they already exist.",
    )
    return parser.parse_args()


def resolve_predict_dir(args: argparse.Namespace) -> Path:
    """Return the resolved cosmos-predict repo path or exit with an error."""
    if args.cosmos_predict_dir is not None:
        return args.cosmos_predict_dir.expanduser().resolve()

    env = os.environ.get("COSMOS_PREDICT_DIR")
    if env:
        return Path(env).expanduser().resolve()

    print(
        "Error: --cosmos-predict-dir not provided and COSMOS_PREDICT_DIR env var not set.",
        file=sys.stderr,
    )
    sys.exit(1)


def load_prompts() -> dict[str, str]:
    """Load and validate raw_prompts.yaml. Returns hazard_type -> prompt mapping."""
    if not PROMPTS_PATH.exists():
        print(
            f"Error: Raw prompts config not found at {PROMPTS_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        data = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        print(f"Error parsing {PROMPTS_PATH}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, dict):
        print(
            f"Error: {PROMPTS_PATH} must contain a YAML mapping of hazard_type -> prompt",
            file=sys.stderr,
        )
        sys.exit(1)

    return {k: str(v).strip() for k, v in data.items()}


def update_manifest(
    output_dir: Path,
    predict_repo: Path,
    resolution: str,
    fps: int,
    duration: float,
    expected_raw: int,
) -> None:
    """Merge the raw provenance block into data/generation_manifest.json."""
    manifest_path = PROJECT_ROOT / "data" / "generation_manifest.json"
    manifest: dict = {}

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}

    if not isinstance(manifest, dict):
        manifest = {}

    total_files = len(list(output_dir.glob("*.mp4")))
    manifest["raw"] = {
        "total_generated": total_files,
        "expected": expected_raw,
        "resolution": resolution,
        "fps": fps,
        "duration_seconds": duration,
        "source": "cosmos-predict (synthetic baseline — no real R6 clips)",
        "predict_repo_path": str(predict_repo),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Updated generation manifest: {manifest_path}")


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir.expanduser().resolve()
    predict_dir = resolve_predict_dir(args)
    script_path = predict_dir / "run_predict.py"

    if not script_path.exists():
        print(
            f"Error: Predict script not found at {script_path}. "
            "Check --cosmos-predict-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    prompts_by_hazard = load_prompts()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_this_run = 0
    skipped = 0
    failed = 0

    # Sort hazard types for deterministic clip naming and ordering
    for hazard_type in sorted(prompts_by_hazard):
        prompt = prompts_by_hazard[hazard_type]
        output_path = output_dir / f"raw_{hazard_type}.mp4"

        # Freeze guard — skip existing files unless --force
        if output_path.exists() and not args.force:
            print(f"Skipping {output_path} (exists)")
            skipped += 1
            continue

        cmd = [
            sys.executable,
            str(script_path),
            "--prompt",
            prompt,
            "--duration",
            str(args.duration),
            "--output",
            str(output_path),
        ]
        print(f"Generating: {output_path.name}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(
                f"Error: generation failed for {output_path.name} "
                f"(exit code {result.returncode})",
                file=sys.stderr,
            )
            failed += 1
            continue

        generated_this_run += 1

    print(
        f"\nRaw generation done. "
        f"Generated: {generated_this_run}, "
        f"Skipped (existing): {skipped}, "
        f"Failed: {failed}"
    )
    expected_raw = len(prompts_by_hazard)
    update_manifest(output_dir, predict_dir, args.resolution, args.fps, args.duration, expected_raw)


if __name__ == "__main__":
    main()

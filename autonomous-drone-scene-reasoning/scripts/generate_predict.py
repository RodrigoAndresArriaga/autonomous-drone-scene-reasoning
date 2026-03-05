"""
Generate Cosmos Predict synthetic clips from taxonomy-aligned prompts.

Prompts are loaded from configs/predict_prompts.yaml. Keys in that file must
be canonical hazard types from reasoning/hazard_schema.py. Each category has
3 prompts, producing:
  5 categories × 3 prompts = 15 clips

Clips are named:  predict_{category}_{1|2|3}.mp4

Generation is skipped by default when an output file already exists.
Use --force to regenerate existing files.

After generation, writes/updates data/generation_manifest.json with the
predict provenance block. Any existing transfer block is preserved so that
the manifest accumulates both sections across separate runs.

Usage:
    python scripts/generate_predict.py \\
        --cosmos-predict-dir /home/ubuntu/cosmos/cosmos-predict

    # Regenerate everything from scratch:
    python scripts/generate_predict.py \\
        --cosmos-predict-dir /home/ubuntu/cosmos/cosmos-predict \\
        --force

    # Custom duration and output dir:
    python scripts/generate_predict.py \\
        --cosmos-predict-dir /home/ubuntu/cosmos/cosmos-predict \\
        --duration 8 \\
        --output-dir /path/to/output
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
PROMPTS_PATH = PROJECT_ROOT / "configs" / "predict_prompts.yaml"

DATASET_VERSION = "v1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Cosmos Predict synthetic clips from taxonomy-aligned prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "predict",
        help="Directory to write Predict clips (default: data/predict).",
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


def load_prompts() -> dict[str, list[str]]:
    """Load and validate predict_prompts.yaml. Returns category -> [prompt, ...] mapping."""
    if not PROMPTS_PATH.exists():
        print(
            f"Error: Predict prompts config not found at {PROMPTS_PATH}",
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
            f"Error: {PROMPTS_PATH} must contain a YAML mapping of category -> [prompts]",
            file=sys.stderr,
        )
        sys.exit(1)

    cleaned: dict[str, list[str]] = {}
    for category, prompts in data.items():
        if not isinstance(prompts, list) or not prompts:
            print(
                f"Error: prompts for category '{category}' must be a non-empty list",
                file=sys.stderr,
            )
            sys.exit(1)
        cleaned[category] = [str(p).strip() for p in prompts]

    return cleaned


def update_manifest(
    output_dir: Path,
    predict_repo: Path,
    resolution: str,
    fps: int,
    duration: float,
    expected_predict: int,
) -> None:
    """Merge the predict provenance block into data/generation_manifest.json.

    Dataset freeze (dataset_version + frozen_at) is only set when both transfer
    and predict blocks exist and their total_generated matches the expected counts,
    ensuring an incomplete dataset is never marked as frozen.
    """
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
    manifest["predict"] = {
        "total_generated": total_files,
        "resolution": resolution,
        "fps": fps,
        "duration_seconds": duration,
        "predict_repo_path": str(predict_repo),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Only freeze when both blocks are present and counts match expectations.
    transfer_ok = manifest.get("transfer", {}).get("total_generated", 0) > 0
    predict_ok = manifest.get("predict", {}).get("total_generated", 0) == expected_predict
    if "transfer" in manifest and "predict" in manifest and transfer_ok and predict_ok:
        manifest.setdefault("dataset_version", DATASET_VERSION)
        manifest.setdefault("frozen_at", datetime.now(timezone.utc).isoformat())
    elif "transfer" in manifest and "predict" in manifest:
        print(
            f"Note: dataset not yet frozen — predict count {total_files}/{expected_predict} "
            f"or transfer block incomplete.",
            file=sys.stderr,
        )

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

    prompts_by_category = load_prompts()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_this_run = 0
    skipped = 0
    failed = 0

    # Sort categories for deterministic clip naming and ordering
    for category in sorted(prompts_by_category):
        prompts = prompts_by_category[category]
        for idx, prompt in enumerate(prompts, start=1):
            output_path = output_dir / f"predict_{category}_{idx}.mp4"

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
        f"\nPredict generation done. "
        f"Generated: {generated_this_run}, "
        f"Skipped (existing): {skipped}, "
        f"Failed: {failed}"
    )
    expected_predict = sum(len(prompts) for prompts in prompts_by_category.values())
    update_manifest(output_dir, predict_dir, args.resolution, args.fps, args.duration, expected_predict)


if __name__ == "__main__":
    main()

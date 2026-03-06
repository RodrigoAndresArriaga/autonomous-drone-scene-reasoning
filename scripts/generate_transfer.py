"""
Generate Cosmos Transfer variants for every baseline clip in data/raw/.

For each .mp4 in --raw-dir, produces 4 domain-shifted variants in --output-dir:
  lowlight, fog, strong_shadow, material_shift

Clips are named:  {stem}_{variant}.mp4
Total output:     5 baseline clips × 4 variants = 20 clips

Generation is skipped by default when an output file already exists.
Use --force to regenerate existing files.

After generation, writes/updates data/generation_manifest.json with the
transfer provenance block so the manifest accumulates both transfer and predict
sections across separate runs.

Usage:
    python scripts/generate_transfer.py \\
        --cosmos-transfer-dir /home/ubuntu/cosmos/cosmos-transfer

    # Regenerate everything from scratch:
    python scripts/generate_transfer.py \\
        --cosmos-transfer-dir /home/ubuntu/cosmos/cosmos-transfer \\
        --force

    # Override default dirs:
    python scripts/generate_transfer.py \\
        --raw-dir /path/to/clips \\
        --output-dir /path/to/output \\
        --cosmos-transfer-dir /home/ubuntu/cosmos/cosmos-transfer
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_VERSION = "v1.0"

# 4 domain-shift variants and their natural language prompts.
# Prompts describe the visual transformation while preserving geometry.
TRANSFER_VARIANTS: dict[str, str] = {
    "lowlight": (
        "Render the same scene with significantly reduced ambient light, "
        "strong low-light conditions, and deep shadows, "
        "while preserving all geometry, obstacles, and hazards exactly."
    ),
    "fog": (
        "Add moderate environmental fog or haze that partially obscures distant "
        "areas but keeps the underlying spatial layout, obstacles, and hazards "
        "identical."
    ),
    "strong_shadow": (
        "Introduce strong directional lighting that creates pronounced shadows "
        "and high contrast across surfaces and objects, without altering geometry "
        "or hazard placement in any way."
    ),
    "material_shift": (
        "Change surface materials and textures on walls, floor, and debris "
        "to a different visual style while keeping geometry, obstacles, and "
        "hazard locations completely identical."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Cosmos Transfer domain-shift variants for baseline clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing baseline .mp4 clips (default: data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "transfer",
        help="Directory to write Transfer variant clips (default: data/transfer).",
    )
    parser.add_argument(
        "--cosmos-transfer-dir",
        type=Path,
        default=None,
        help=(
            "Path to the cosmos-transfer repo on this machine. "
            "Falls back to COSMOS_TRANSFER_DIR env var if not provided."
        ),
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


def resolve_transfer_dir(args: argparse.Namespace) -> Path:
    """Return the resolved cosmos-transfer repo path or exit with an error."""
    if args.cosmos_transfer_dir is not None:
        return args.cosmos_transfer_dir.expanduser().resolve()

    env = os.environ.get("COSMOS_TRANSFER_DIR")
    if env:
        return Path(env).expanduser().resolve()

    print(
        "Error: --cosmos-transfer-dir not provided and COSMOS_TRANSFER_DIR env var not set.",
        file=sys.stderr,
    )
    sys.exit(1)


def update_manifest(
    output_dir: Path,
    transfer_repo: Path,
    resolution: str,
    fps: int,
    expected_transfer: int,
) -> None:
    """Merge the transfer provenance block into data/generation_manifest.json.

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
    manifest["transfer"] = {
        "variants_per_clip": len(TRANSFER_VARIANTS),
        "total_generated": total_files,
        "resolution": resolution,
        "fps": fps,
        "transfer_repo_path": str(transfer_repo),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Only freeze when both blocks are present and counts match expectations.
    transfer_ok = manifest.get("transfer", {}).get("total_generated", 0) == expected_transfer
    predict_ok = manifest.get("predict", {}).get("total_generated", 0) > 0
    if "transfer" in manifest and "predict" in manifest and transfer_ok and predict_ok:
        manifest.setdefault("dataset_version", DATASET_VERSION)
        manifest.setdefault("frozen_at", datetime.now(timezone.utc).isoformat())
    elif "transfer" in manifest and "predict" in manifest:
        print(
            f"Note: dataset not yet frozen — transfer count {total_files}/{expected_transfer} "
            f"or predict block incomplete.",
            file=sys.stderr,
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Updated generation manifest: {manifest_path}")


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()
    transfer_dir = resolve_transfer_dir(args)
    script_path = transfer_dir / "run_transfer.py"

    if not script_path.exists():
        print(
            f"Error: Transfer script not found at {script_path}. "
            "Check --cosmos-transfer-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not raw_dir.exists():
        print(f"Error: Raw clip directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    input_clips = sorted(raw_dir.glob("*.mp4"))
    if not input_clips:
        print(f"Warning: No .mp4 files found in {raw_dir}")

    generated_this_run = 0
    skipped = 0
    failed = 0

    for input_path in input_clips:
        for variant_name, prompt in TRANSFER_VARIANTS.items():
            output_path = output_dir / f"{input_path.stem}_{variant_name}.mp4"

            # Freeze guard — skip existing files unless --force
            if output_path.exists() and not args.force:
                print(f"Skipping {output_path} (exists)")
                skipped += 1
                continue

            cmd = [
                sys.executable,
                str(script_path),
                "--input",
                str(input_path),
                "--prompt",
                prompt,
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
        f"\nTransfer generation done. "
        f"Generated: {generated_this_run}, "
        f"Skipped (existing): {skipped}, "
        f"Failed: {failed}"
    )
    expected_transfer = len(input_clips) * len(TRANSFER_VARIANTS)
    update_manifest(output_dir, transfer_dir, args.resolution, args.fps, expected_transfer)


if __name__ == "__main__":
    main()

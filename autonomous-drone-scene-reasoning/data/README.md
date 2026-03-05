# Dataset Layout

This directory holds all video clips and ground-truth labels for the 40-clip benchmark.

```
data/
├── raw/               # 5 synthetic baseline clips (Phase 1, from Cosmos Predict)
├── transfer/          # 20 domain-shifted clips   (Phase 2, from Cosmos Transfer)
├── predict/           # 15 synthetic clips        (Phase 3, from Cosmos Predict)
├── metadata.csv       # Ground-truth labels for all 40 clips
└── generation_manifest.json  # Provenance record written by generation scripts
```

## Subdirectories

| Directory | Content | How populated |
|-----------|---------|---------------|
| `raw/` | 5 synthetic baseline clips (one per hazard category) | `scripts/generate_raw.py` |
| `transfer/` | 20 Transfer variants (4 domain shifts per baseline clip) | `scripts/generate_transfer.py` |
| `predict/` | 15 Predict synthetic clips (3 per hazard category) | `scripts/generate_predict.py` |

**Note on raw/ tier:** No real R6 clips are used. The 5 baseline clips are generated via
Cosmos Predict using single-path corridor prompts defined in `configs/raw_prompts.yaml`.
Each prompt depicts a narrow one-way passage where a canonical hazard occupies the full
route width with no bypass option. Cosmos Transfer then applies 4 domain-shift variants
(lowlight, fog, strong_shadow, material_shift) to each of these baseline clips.

**Execution order:**
```
1. python scripts/generate_raw.py      --cosmos-predict-dir <path>    → data/raw/   (5 clips)
2. python scripts/generate_transfer.py --cosmos-transfer-dir <path>   → data/transfer/ (20 clips)
3. python scripts/generate_predict.py  --cosmos-predict-dir <path>    → data/predict/  (15 clips)
4. Populate data/metadata.csv with ground-truth labels for all 40 clips
5. python scripts/run_scenarios.py data/raw/*.mp4 data/transfer/*.mp4 data/predict/*.mp4
```

## Single-path constraint

All 40 clips are designed around a **single-path corridor** constraint:
- The hazard occupies the **full width** of the only available route.
- There are no side exits, branch paths, or room to bypass the hazard laterally.
- The agent must classify the path as safe to proceed or unsafe and stop/reroute.

This constraint ensures classification labels are unambiguous and evaluation is meaningful:
a model that outputs "safe" when the only path is blocked by a hazard is clearly wrong.

## metadata.csv columns

| Column | Description |
|--------|-------------|
| `clip_id` | Filename stem (e.g. `raw_debris`, `raw_debris_lowlight`, `predict_debris_1`) |
| `tier` | `raw` \| `transfer` \| `predict` |
| `class_label` | Safety class: `A` (safe), `B` (caution), `C` (unsafe) |
| `drone_safe` | `true` \| `false` — drone path classification |
| `human_safe` | `true` \| `false` — human-follow path classification |
| `primary_hazard` | Canonical hazard type from `reasoning/hazard_schema.py` |

**Labeling rules:**
- `raw` — predefined by hazard type in `configs/raw_prompts.yaml` (single hazard per clip).
- `transfer` — inherited from the corresponding baseline clip (geometry preserved by Transfer).
- `predict` — manual review using the affordance matrix; categories from `configs/predict_prompts.yaml`.

## generation_manifest.json

Written automatically by the generation scripts after each run. Contains provenance for all three generation phases:

```json
{
  "dataset_version": "v1.0",
  "frozen_at": "2026-...",
  "raw": {
    "total_generated": 5,
    "expected": 5,
    "resolution": "768x512",
    "fps": 4,
    "duration_seconds": 6.0,
    "source": "cosmos-predict (synthetic baseline — no real R6 clips)",
    "predict_repo_path": "/home/ubuntu/cosmos/cosmos-predict",
    "timestamp": "2026-..."
  },
  "transfer": {
    "variants_per_clip": 4,
    "total_generated": 20,
    "resolution": "768x512",
    "fps": 4,
    "transfer_repo_path": "/home/ubuntu/cosmos/cosmos-transfer",
    "timestamp": "2026-..."
  },
  "predict": {
    "total_generated": 15,
    "resolution": "768x512",
    "fps": 4,
    "duration_seconds": 6.0,
    "predict_repo_path": "/home/ubuntu/cosmos/cosmos-predict",
    "timestamp": "2026-..."
  }
}
```

Each script only writes/updates its own block and preserves the others. `dataset_version` and
`frozen_at` are set automatically the first time both transfer and predict phases are complete,
and are never overwritten on subsequent runs.

## Dataset freeze

After all clips are generated and `metadata.csv` is populated:

- Do **not** regenerate clips.
- Do **not** modify clips.
- Run the benchmark only against the frozen 40-clip set.

Use `--force` on the generation scripts only if intentional regeneration is required.

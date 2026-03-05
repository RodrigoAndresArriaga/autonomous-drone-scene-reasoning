"""
Compute benchmark metrics from scenario_rollup.jsonl vs data/metadata.csv.

Reads:
  data/metadata.csv             — ground-truth labels (clip_id, tier, class_label,
                                  drone_safe, human_safe, primary_hazard)
  outputs/scenario_rollup.jsonl — pipeline predictions from run_scenarios.py

For clips evaluated in rolling mode (multiple rows per clip), the strictest
recommendation across all windows is used (C > B > A).

Computes and prints:
  - Per-class (A / B / C) precision, recall, F1
  - Macro F1
  - Confusion matrix
  - Unsafe guidance rate  (pipeline said A but truth is B or C — most dangerous error)
  - False safe drone rate (pipeline said drone_safe=True but truth says False)
  - Coverage              (fraction of ground-truth clips found in the rollup)

Writes results to outputs/metrics_report.json for programmatic use.

Usage:
    python scripts/compute_metrics.py

    python scripts/compute_metrics.py \\
        --metadata data/metadata.csv \\
        --rollup   outputs/scenario_rollup.jsonl \\
        --output   outputs/metrics_report.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Recommendation string → predicted class label
# ---------------------------------------------------------------------------
# Maps every value in safety/recommendation.py RECOMMENDATIONS to A / B / C.
_REC_TO_CLASS: dict[str, str] = {
    "Proceed and guide human": "A",
    "Proceed but do not guide": "B",
    "Hold position": "B",
    "Scout forward before guiding human": "B",
    "Reroute before guiding": "C",
}
# Fallback for any unrecognised recommendation string
_DEFAULT_CLASS = "B"

# Class ordering for "strictest" aggregation over rolling windows
_CLASS_RANK: dict[str, int] = {"A": 0, "B": 1, "C": 2}


def _rec_to_class(rec: str) -> str:
    for key, cls in _REC_TO_CLASS.items():
        if key.lower() in rec.lower():
            return cls
    return _DEFAULT_CLASS


def _predicted_drone_safe(cls: str) -> bool:
    """Drone is considered safe if pipeline did NOT say Reroute (class C)."""
    return cls != "C"


def _predicted_human_safe(cls: str) -> bool:
    """Human is considered safe only if pipeline said Proceed and guide (class A)."""
    return cls == "A"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_metadata(path: Path) -> dict[str, dict]:
    """Return mapping clip_id -> ground-truth row dict."""
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            row = dict(zip(header, parts))
            cid = row["clip_id"]
            row["drone_safe"] = row["drone_safe"].strip().lower() == "true"
            row["human_safe"] = row["human_safe"].strip().lower() == "true"
            rows[cid] = row
    return rows


def load_rollup(path: Path) -> dict[str, str]:
    """Return mapping clip_id -> strictest predicted class across all windows."""
    # Accumulate rows per scenario_id
    per_clip: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("scenario_id", "")
            rec = row.get("recommendation") or ""
            if isinstance(rec, dict):
                rec = rec.get("recommendation", "")
            per_clip[sid].append(_rec_to_class(rec))

    # Take strictest class per clip (C > B > A)
    result: dict[str, str] = {}
    for sid, classes in per_clip.items():
        result[sid] = max(classes, key=lambda c: _CLASS_RANK.get(c, 0))
    return result


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_metrics(
    ground_truth: dict[str, dict],
    predictions: dict[str, str],
) -> dict:
    classes = ["A", "B", "C"]

    # Align: only clips present in both
    common = sorted(set(ground_truth) & set(predictions))
    missing_from_rollup = sorted(set(ground_truth) - set(predictions))
    extra_in_rollup = sorted(set(predictions) - set(ground_truth))

    truth_classes = [ground_truth[c]["class_label"] for c in common]
    pred_classes = [predictions[c] for c in common]

    # Confusion matrix: rows=truth, cols=pred
    conf: dict[str, dict[str, int]] = {t: {p: 0 for p in classes} for t in classes}
    for t, p in zip(truth_classes, pred_classes):
        if t in conf and p in classes:
            conf[t][p] += 1

    # Per-class P/R/F1
    per_class: dict[str, dict] = {}
    for cls in classes:
        tp = conf[cls][cls]
        fp = sum(conf[other][cls] for other in classes if other != cls)
        fn = sum(conf[cls][other] for other in classes if other != cls)
        p, r, f1 = _precision_recall_f1(tp, fp, fn)
        per_class[cls] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                          "support": truth_classes.count(cls)}

    macro_f1 = round(sum(v["f1"] for v in per_class.values()) / len(classes), 4)
    accuracy = round(sum(t == p for t, p in zip(truth_classes, pred_classes)) / len(common), 4) if common else 0.0

    # Unsafe guidance rate: pipeline says A but truth is B or C
    unsafe_guidance = [
        c for c in common
        if predictions[c] == "A" and ground_truth[c]["class_label"] in ("B", "C")
    ]
    unsafe_guidance_rate = round(len(unsafe_guidance) / len(common), 4) if common else 0.0

    # False safe drone rate: pipeline says drone_safe=True but truth says False
    false_safe_drone = [
        c for c in common
        if _predicted_drone_safe(predictions[c]) and not ground_truth[c]["drone_safe"]
    ]
    false_safe_drone_rate = round(len(false_safe_drone) / len(common), 4) if common else 0.0

    # Per-tier breakdown
    tiers = ["raw", "transfer", "predict"]
    tier_accuracy: dict[str, float] = {}
    for tier in tiers:
        tier_clips = [c for c in common if ground_truth[c].get("tier") == tier]
        if tier_clips:
            correct = sum(predictions[c] == ground_truth[c]["class_label"] for c in tier_clips)
            tier_accuracy[tier] = round(correct / len(tier_clips), 4)
        else:
            tier_accuracy[tier] = None

    return {
        "coverage": {
            "evaluated": len(common),
            "ground_truth_total": len(ground_truth),
            "missing_from_rollup": missing_from_rollup,
            "extra_in_rollup": extra_in_rollup,
        },
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": conf,
        "unsafe_guidance_rate": unsafe_guidance_rate,
        "unsafe_guidance_clips": unsafe_guidance,
        "false_safe_drone_rate": false_safe_drone_rate,
        "false_safe_drone_clips": false_safe_drone,
        "tier_accuracy": tier_accuracy,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def format_report(m: dict) -> str:
    cov = m["coverage"]
    lines = [
        "=" * 60,
        "  BENCHMARK METRICS",
        "=" * 60,
        f"  Coverage   : {cov['evaluated']} / {cov['ground_truth_total']} clips evaluated",
        f"  Accuracy   : {m['accuracy']:.1%}",
        f"  Macro F1   : {m['macro_f1']:.4f}",
        "",
        "  Per-class metrics:",
        f"  {'Class':<6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}",
        "  " + "-" * 36,
    ]
    for cls, v in m["per_class"].items():
        lines.append(
            f"  {cls:<6} {v['precision']:>6.3f} {v['recall']:>6.3f} {v['f1']:>6.3f} {v['support']:>8}"
        )

    lines += [
        "",
        "  Confusion matrix (rows=truth, cols=predicted):",
        f"  {'':>6} {'pred A':>8} {'pred B':>8} {'pred C':>8}",
    ]
    conf = m["confusion_matrix"]
    for truth_cls in ["A", "B", "C"]:
        row = conf.get(truth_cls, {})
        lines.append(
            f"  {'true ' + truth_cls:>6} {row.get('A', 0):>8} {row.get('B', 0):>8} {row.get('C', 0):>8}"
        )

    ugr = m["unsafe_guidance_rate"]
    fsd = m["false_safe_drone_rate"]
    lines += [
        "",
        "  Safety-critical rates:",
        f"  Unsafe guidance rate   : {ugr:.1%}  {_bar(ugr)}  (predicted A, truth B/C)",
        f"  False safe drone rate  : {fsd:.1%}  {_bar(fsd)}  (predicted drone safe, truth not)",
        "",
        "  Per-tier accuracy:",
    ]
    for tier, acc in m["tier_accuracy"].items():
        if acc is not None:
            lines.append(f"  {tier:<10} : {acc:.1%}")
        else:
            lines.append(f"  {tier:<10} : — (no clips)")

    if cov["missing_from_rollup"]:
        lines += ["", f"  ⚠ {len(cov['missing_from_rollup'])} ground-truth clips not in rollup:"]
        for c in cov["missing_from_rollup"][:10]:
            lines.append(f"      {c}")
        if len(cov["missing_from_rollup"]) > 10:
            lines.append(f"      ... and {len(cov['missing_from_rollup']) - 10} more")

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute benchmark metrics from scenario rollup vs ground-truth metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "metadata.csv",
        help="Path to ground-truth metadata CSV (default: data/metadata.csv).",
    )
    parser.add_argument(
        "--rollup",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "scenario_rollup.jsonl",
        help="Path to scenario rollup JSONL (default: outputs/scenario_rollup.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics_report.json",
        help="Path to write JSON metrics report (default: outputs/metrics_report.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.metadata.exists():
        print(f"Error: metadata file not found: {args.metadata}", file=sys.stderr)
        sys.exit(1)
    if not args.rollup.exists():
        print(f"Error: rollup file not found: {args.rollup}", file=sys.stderr)
        sys.exit(1)

    ground_truth = load_metadata(args.metadata)
    predictions = load_rollup(args.rollup)

    if not ground_truth:
        print("Error: metadata.csv has no rows.", file=sys.stderr)
        sys.exit(1)
    if not predictions:
        print("Error: scenario_rollup.jsonl has no rows.", file=sys.stderr)
        sys.exit(1)

    metrics = compute_metrics(ground_truth, predictions)
    print(format_report(metrics))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMetrics written to: {args.output}")


if __name__ == "__main__":
    main()

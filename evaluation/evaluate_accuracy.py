import argparse
import csv
import json
from pathlib import Path

import numpy as np

REQUIRED = ("image", "true_long_mm", "true_short_mm", "pred_long_mm", "pred_short_mm")


def main():
    parser = argparse.ArgumentParser(description="Calculate honest per-image and summary measurement errors from a CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with the columns shown in docs/accuracy_template.csv")
    parser.add_argument("--out", type=Path, default=Path("outputs/accuracy_report.json"))
    args = parser.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    rows = []
    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [key for key in REQUIRED if key not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"CSV missing columns: {', '.join(missing)}")
        for line, row in enumerate(reader, start=2):
            try:
                tl, ts = float(row["true_long_mm"]), float(row["true_short_mm"])
                pl, ps = float(row["pred_long_mm"]), float(row["pred_short_mm"])
                if min(tl, ts, pl, ps) <= 0:
                    raise ValueError("measurements must be positive")
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"Invalid numeric data on CSV line {line}: {exc}")
            rows.append({"image": row["image"], "true_long_mm": tl, "true_short_mm": ts,
                         "pred_long_mm": pl, "pred_short_mm": ps,
                         "abs_error_long_mm": abs(pl - tl), "abs_error_short_mm": abs(ps - ts)})
    if not rows:
        raise SystemExit("CSV has a header but no measurement rows")

    long_errors = np.array([r["abs_error_long_mm"] for r in rows])
    short_errors = np.array([r["abs_error_short_mm"] for r in rows])
    report = {
        "n_images": len(rows),
        "long_side_abs_error_mm": {"mae": float(np.mean(long_errors)), "median": float(np.median(long_errors)), "worst": float(np.max(long_errors))},
        "short_side_abs_error_mm": {"mae": float(np.mean(short_errors)), "median": float(np.median(short_errors)), "worst": float(np.max(short_errors))},
        "per_image": rows,
        "note": "Only entered measurements are evaluated; this report does not imply a trained model.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluated {len(rows)} images")
    print(f"Long side MAE/median/worst: {np.mean(long_errors):.2f}/{np.median(long_errors):.2f}/{np.max(long_errors):.2f} mm")
    print(f"Short side MAE/median/worst: {np.mean(short_errors):.2f}/{np.median(short_errors):.2f}/{np.max(short_errors):.2f} mm")
    print(f"Report saved: {args.out}")


if __name__ == "__main__":
    main()

"""
sort.py

Takes the output of classify.py (photos pooled flat by class, plus
classification_log.csv) and reorganizes them into the per-tower MIRRORED
structure:

    output/tower_22/rgb/Phase/DJI_..._0001_V.JPG        <- original filename restored
    output/tower_22/rgb/Top/DJI_..._0017_V.JPG
    output/tower_22/rgb/Earth_wire/DJI_..._0016_V.JPG
    output/tower_29/rgb/Phase/DJI_..._0034_V.JPG
    ...
    output/tower_22/rgb/predictions_log.csv   <- per-tower log slice
    ...

No model, no feature extraction -- this only reads classify.py's already
computed predictions from the log and copies files accordingly. --input
(classify.py's output folder) is only ever READ from, never modified.

Usage:
    python sort.py --input classified --output sorted_by_tower --dry_run
    python sort.py --input classified --output sorted_by_tower
"""

import argparse
import csv
import shutil
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Reorganize classify.py's pooled output into a per-tower mirrored structure.")
    parser.add_argument("--input", type=str, required=True,
                         help="Path to classify.py's output folder (contains classification_log.csv "
                              "and the pooled class folders). Read-only.")
    parser.add_argument("--output", type=str, required=True,
                         help="Path to a NEW folder for the per-tower mirrored structure.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.is_dir():
        raise FileNotFoundError(f"--input path does not exist or is not a folder: {input_root}")
    if input_root.resolve() == output_root.resolve():
        raise ValueError("--output must be a different folder from --input.")

    log_path = input_root / "classification_log.csv"
    if not log_path.is_file():
        raise FileNotFoundError(
            f"classification_log.csv not found at {log_path}. "
            f"Did you run classify.py first, and point --input at ITS --output folder?"
        )

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} entries from {log_path}")

    per_tower_rows = defaultdict(list)   # tower_name -> list of rows, for per-tower logs
    grand_total = {}
    missing_files = []

    for row in rows:
        tower = row["tower"]
        original_filename = row["original_filename"]
        sorted_into = row["sorted_into"]
        pooled_filename = row["pooled_filename"]

        # Source: where classify.py actually put the file (pooled, flat by class)
        src_path = input_root / sorted_into / pooled_filename

        # Destination: per-tower mirrored structure, ORIGINAL filename restored
        dest_dir = output_root / tower / "rgb" / sorted_into
        dest_path = dest_dir / original_filename

        if not src_path.exists():
            missing_files.append(src_path)
            continue

        if args.dry_run:
            print(f"  {src_path}  ->  {dest_path}")
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            if dest_path.exists():
                print(f"  [warning] already exists, skipping: {dest_path}")
            else:
                shutil.copy2(str(src_path), str(dest_path))  # always copy -- input stays untouched

        grand_total[sorted_into] = grand_total.get(sorted_into, 0) + 1
        per_tower_rows[tower].append([original_filename, row["predicted_class"], row["confidence"], sorted_into])

    print(f"\n=== Grand total across all towers ===")
    print(grand_total)

    if missing_files:
        print(f"\n[warning] {len(missing_files)} file(s) referenced in the log were not found "
              f"in --input (was the pooled folder modified or incomplete?):")
        for p in missing_files[:10]:
            print(f"  {p}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    if not args.dry_run:
        for tower, tower_rows in per_tower_rows.items():
            tower_log_dir = output_root / tower / "rgb"
            tower_log_dir.mkdir(parents=True, exist_ok=True)
            tower_log_path = tower_log_dir / "predictions_log.csv"
            with open(tower_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "predicted_class", "confidence", "sorted_into"])
                writer.writerows(tower_rows)
        print(f"\nResults written to: {output_root.resolve()}")
        print(f"  - Per tower: {output_root}/tower_X/rgb/Phase, /Top, /Earth_wire, etc.")
        print(f"Input (classify.py's output) untouched: {input_root.resolve()}")


if __name__ == "__main__":
    main()

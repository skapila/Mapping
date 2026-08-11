"""
build_training_dataset.py

Pools images ACROSS all towers, by class, into one flat class-organized
dataset -- the exact folder shape train_layer1.py expects (Section 8 of
the report):

    data/
      Phase/       <- every Phase photo from every tower
      Top/         <- every Top photo from every tower
      Earth_wire/  <- every Earth_wire photo from every tower
      Bottom/      <- every Bottom photo from every tower (empty for now)

Reads from your existing per-tower organized folder:
    Result_data_organized/tower_1/rgb/Phase/*.JPG
    Result_data_organized/tower_1/rgb/Top/*.JPG
    Result_data_organized/tower_2/rgb/Phase/*.JPG
    ...

--root is only ever READ from -- nothing is modified there. --output is a
brand new folder. Filenames are prefixed with their tower name to avoid
collisions when pooling across towers (e.g. tower_1_DJI_..._0034_V.JPG).

Usage:
    # 1. Dry run first -- prints the plan, touches nothing.
    python build_training_dataset.py --root Result_data_organized --output data --dry_run

    # 2. Once the plan looks right, actually copy:
    python build_training_dataset.py --root Result_data_organized --output data
"""

import argparse
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["Phase", "Top", "Earth_wire", "Bottom"]


def main():
    parser = argparse.ArgumentParser(description="Pool per-tower class folders into one flat class-organized dataset.")
    parser.add_argument("--root", type=str, required=True,
                         help="Path to the organized-by-tower folder (contains tower_1, tower_2, ...).")
    parser.add_argument("--output", type=str, required=True,
                         help="Path to a NEW folder where data/Phase, data/Top, etc. will be created.")
    parser.add_argument("--move", action="store_true",
                         help="Move files instead of copying (default: copy, --root stays intact).")
    parser.add_argument("--dry_run", action="store_true",
                         help="Print the plan without touching any files.")
    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path(args.output)

    if not root.is_dir():
        raise FileNotFoundError(f"--root path does not exist or is not a folder: {root}")
    if root.resolve() == output_root.resolve():
        raise ValueError("--output must be a different folder from --root.")

    tower_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("tower_"))
    if not tower_dirs:
        raise ValueError(f"No tower_* folders found under {root}")

    if not args.dry_run:
        total_src_bytes = sum(
            p.stat().st_size
            for tower_dir in tower_dirs
            for class_name in CLASS_NAMES
            for p in (tower_dir / "rgb" / class_name).glob("*")
            if (tower_dir / "rgb" / class_name).is_dir() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        free_bytes = shutil.disk_usage(output_root.parent if output_root.parent.exists() else root).free
        print(f"Total source data size: {total_src_bytes / 1e9:.2f} GB")
        print(f"Free disk space at destination: {free_bytes / 1e9:.2f} GB")
        if free_bytes < total_src_bytes * 1.1:  # small safety margin
            print("[WARNING] Free disk space is close to (or less than) the data size being copied. "
                  "This is a common cause of 0-byte / truncated files. Consider freeing up space "
                  "or using --move instead of copy before proceeding.\n")

    grand_total = {cls: 0 for cls in CLASS_NAMES}
    size_mismatches = []

    for class_name in CLASS_NAMES:
        dest_dir = output_root / class_name
        if not args.dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for tower_dir in tower_dirs:
            src_class_dir = tower_dir / "rgb" / class_name
            if not src_class_dir.is_dir():
                continue

            images = sorted(p for p in src_class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            for src_path in images:
                dest_name = f"{tower_dir.name}_{src_path.name}"
                dest_path = dest_dir / dest_name

                if args.dry_run:
                    print(f"  [{class_name}] {src_path}  ->  {dest_path}")
                else:
                    if dest_path.exists():
                        print(f"  [warning] already exists, skipping: {dest_path}")
                        continue
                    src_size = src_path.stat().st_size
                    if src_size == 0:
                        print(f"  [warning] SOURCE file is already 0 bytes, skipping "
                              f"(corruption predates this script): {src_path}")
                        continue
                    if args.move:
                        shutil.move(str(src_path), str(dest_path))
                    else:
                        shutil.copy2(str(src_path), str(dest_path))
                    # Verify the copy actually landed correctly -- catches disk-full
                    # or interrupted-copy situations immediately instead of silently
                    # producing a 0-byte or truncated file that only fails much later
                    # (e.g. inside Step 4's feature extraction).
                    dest_size = dest_path.stat().st_size
                    if dest_size != src_size:
                        size_mismatches.append((src_path, dest_path, src_size, dest_size))
                        print(f"  [ERROR] size mismatch after copy! src={src_size} bytes, "
                              f"dest={dest_size} bytes: {dest_path}")
                count += 1

        grand_total[class_name] = count
        print(f"{class_name}: {count} images pooled across {len(tower_dirs)} towers"
              f"{'  (EMPTY -- no source images found)' if count == 0 else ''}")

    verb = "Would copy/move" if args.dry_run else ("Moved" if args.move else "Copied")
    print(f"\n{verb} totals: {grand_total}")
    if not args.dry_run:
        print(f"Class-organized dataset written to: {output_root.resolve()}")
        print(f"Source folder untouched: {root.resolve()}" + (" (except files removed by --move)" if args.move else ""))

        if size_mismatches:
            print(f"\n{len(size_mismatches)} file(s) copied with a SIZE MISMATCH (likely corrupted/truncated copies):")
            for src, dest, src_size, dest_size in size_mismatches:
                print(f"  {dest}  (expected {src_size} bytes, got {dest_size} bytes)")
            print("Re-run the script for just these files, or check disk space, then retry.")
        else:
            print("\nAll copied files verified -- destination sizes match source sizes exactly.")


if __name__ == "__main__":
    main()

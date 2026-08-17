"""
reorganize_tower_folders.py

Reads from --root (your existing Result_data folder, untouched) and builds
a BRAND NEW folder tree at --output, one subfolder per tower, each with a
flat 4-category layout:

    <output>/tower_X/rgb/
      Phase/       <- every image from circuit_1/*-phase AND circuit_2/*-phase
      Top/         <- every image from tower_centre
      Earth_wire/  <- every image from earthwire
      Bottom/      <- every image from a 'bottom' folder (if present; created
                       empty otherwise, since the source tree may not have
                       this data yet)

--root is only ever READ from -- nothing is written back into it, and
nothing under --root is deleted or modified, regardless of --move (--move
only affects whether files are copied or moved INTO --output; see below).

Original filenames are preserved exactly (no prefixing). This is a
per-tower operation -- tower_1's Phase folder and tower_2's Phase folder
stay separate; nothing is merged across towers.

Because filenames are kept as-is, a collision is possible if, within the
SAME tower, circuit_1 and circuit_2 happen to contain a file with the
identical name. This is detected and reported rather than silently
overwritten -- you resolve those manually.

Usage:
    # 1. Dry run first -- prints the plan, touches nothing.
    python reorganize_tower_folders.py --root Result_data --output Result_data_organized --dry_run

    # 2. Once the plan looks right, actually copy (Result_data stays untouched):
    python reorganize_tower_folders.py --root Result_data --output Result_data_organized

    # Move instead of copy -- files are REMOVED from Result_data and placed
    # in --output. Only use this once you've verified a copy run looks right.
    python reorganize_tower_folders.py --root Result_data --output Result_data_organized --move
"""

import argparse
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PHASE_FOLDER_NAMES = {"B-phase", "R-phase", "Y-phase"}
CIRCUIT_FOLDER_NAMES = {"circuit_1", "circuit_2"}
TOP_SOURCE_NAMES = {"tower_centre"}
EARTHWIRE_SOURCE_NAMES = {"earthwire"}
BOTTOM_SOURCE_NAMES = {"bottom", "main_leg_bottom"}  # tolerate a couple of likely spellings


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def plan_for_tower(rgb_dir: Path):
    """
    Returns a dict: {"Phase": [paths], "Top": [paths], "Earth_wire": [paths], "Bottom": [paths]}
    describing what should be collected from this tower's rgb/ folder.
    """
    plan = {"Phase": [], "Top": [], "Earth_wire": [], "Bottom": []}

    for child in sorted(rgb_dir.iterdir()):
        if not child.is_dir():
            continue

        if child.name in CIRCUIT_FOLDER_NAMES:
            for phase_dir in sorted(child.iterdir()):
                if phase_dir.is_dir() and phase_dir.name in PHASE_FOLDER_NAMES:
                    plan["Phase"].extend(list_images(phase_dir))

        elif child.name in TOP_SOURCE_NAMES:
            plan["Top"].extend(list_images(child))

        elif child.name in EARTHWIRE_SOURCE_NAMES:
            plan["Earth_wire"].extend(list_images(child))

        elif child.name in BOTTOM_SOURCE_NAMES:
            plan["Bottom"].extend(list_images(child))

        # Anything else under rgb/ (e.g. an already-created Phase/Top/... from
        # a previous run) is intentionally ignored, so re-running is safe.

    return plan


def apply_plan(tower_name, output_rgb_dir, plan, move, dry_run):
    counts = {"Phase": 0, "Top": 0, "Earth_wire": 0, "Bottom": 0}
    collisions = []

    for category, src_paths in plan.items():
        dest_dir = output_rgb_dir / category
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        seen_names = set()
        for src_path in src_paths:
            dest_path = dest_dir / src_path.name  # filename kept identical

            if src_path.name in seen_names or dest_path.exists():
                collisions.append((category, src_path, dest_path))
                continue
            seen_names.add(src_path.name)

            if dry_run:
                print(f"  [{tower_name}] {category}: {src_path}  ->  {dest_path}")
            else:
                if move:
                    shutil.move(str(src_path), str(dest_path))
                else:
                    shutil.copy2(str(src_path), str(dest_path))
            counts[category] += 1

        if category == "Bottom" and not src_paths and not dry_run:
            # No bottom-labeled source data found for this tower -- still
            # create the (empty) folder so the 4-category layout is consistent.
            dest_dir.mkdir(parents=True, exist_ok=True)

    return counts, collisions


def main():
    parser = argparse.ArgumentParser(description="Reorganize each tower's rgb/ folder into Phase/Top/Earth_wire/Bottom.")
    parser.add_argument("--root", type=str, required=True,
                         help="Path to the EXISTING Result_data folder (contains tower_1, tower_2, ...). "
                              "This folder is only read from, never modified.")
    parser.add_argument("--output", type=str, required=True,
                         help="Path to a NEW folder where the reorganized tower_X/rgb/Phase|Top|Earth_wire|Bottom "
                              "structure will be created. Created automatically if it doesn't exist.")
    parser.add_argument("--move", action="store_true",
                         help="Move files out of --root into --output instead of copying "
                              "(default: copy, --root stays fully intact either way in terms of its folder tree).")
    parser.add_argument("--dry_run", action="store_true",
                         help="Print the plan without touching any files.")
    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path(args.output)

    if not root.is_dir():
        raise FileNotFoundError(f"--root path does not exist or is not a folder: {root}")
    if root.resolve() == output_root.resolve():
        raise ValueError("--output must be a different folder from --root (cannot reorganize in place).")

    tower_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("tower_"))
    if not tower_dirs:
        raise ValueError(f"No tower_* folders found directly under {root}")

    grand_total = {"Phase": 0, "Top": 0, "Earth_wire": 0, "Bottom": 0}
    all_collisions = []

    for tower_dir in tower_dirs:
        rgb_dir = tower_dir / "rgb"
        if not rgb_dir.is_dir():
            print(f"[skip] {tower_dir.name}: no 'rgb' subfolder")
            continue

        output_rgb_dir = output_root / tower_dir.name / "rgb"
        plan = plan_for_tower(rgb_dir)
        counts, collisions = apply_plan(tower_dir.name, output_rgb_dir, plan, args.move, args.dry_run)

        for k in grand_total:
            grand_total[k] += counts[k]
        all_collisions.extend(collisions)

        print(f"{tower_dir.name}: Phase={counts['Phase']}  Top={counts['Top']}  "
              f"Earth_wire={counts['Earth_wire']}  Bottom={counts['Bottom']}"
              f"{'  (BOTTOM source not found -- empty folder)' if counts['Bottom'] == 0 else ''}")

    verb = "Would copy/move" if args.dry_run else ("Moved" if args.move else "Copied")
    print(f"\n{verb} totals across all towers: {grand_total}")
    if not args.dry_run:
        print(f"New organized structure written to: {output_root.resolve()}")
        print(f"Original folder untouched: {root.resolve()}" + (" (except files removed by --move)" if args.move else ""))

    if all_collisions:
        print(f"\n{len(all_collisions)} filename collision(s) skipped (same filename already present "
              f"at the destination -- resolve manually, since filenames are kept unchanged):")
        for category, src, dest in all_collisions:
            print(f"  [{category}] {src}  (would have overwritten {dest})")


if __name__ == "__main__":
    main()

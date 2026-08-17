"""
classify.py

Uses the ALREADY TRAINED Layer 1 model (from Step 5) to classify unlabeled
photos across all towers, and stores results POOLED FLAT BY CLASS (not
per-tower) in --output:

    output/Phase/tower_22_DJI_..._0001_V.JPG
    output/Top/tower_22_DJI_..._0017_V.JPG
    output/Earth_wire/tower_22_DJI_..._0016_V.JPG
    output/Phase/tower_29_DJI_..._0034_V.JPG
    ...
    output/classification_log.csv   <- every prediction, with tower + original filename recorded

--root is only ever READ from -- never modified. Every photo is COPIED
(never moved) into --output.

The CSV log records enough info (tower name + original filename) for
sort.py to later reconstruct the per-tower mirrored structure WITHOUT
re-running the model.

Usage:
    python classify.py --root Test_data --model step5_layer1_model.joblib --output classified --dry_run
    python classify.py --root Test_data --model step5_layer1_model.joblib --output classified
"""

import argparse
import csv
import shutil
from pathlib import Path

import joblib
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_backbone(device="cpu"):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def extract_feature(model, transform, image_path, device="cpu"):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(tensor)
    return feature.squeeze(0).cpu().numpy()


def list_flat_images(rgb_dir):
    return sorted(
        p for p in rgb_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser(description="Classify unlabeled photos, pooled flat by class across all towers.")
    parser.add_argument("--root", type=str, required=True,
                         help="Path to the unlabeled tower data (contains tower_22, tower_23, ...). Read-only.")
    parser.add_argument("--model", type=str, required=True,
                         help="Path to the trained model bundle (.joblib) from Step 5.")
    parser.add_argument("--output", type=str, required=True,
                         help="Path to a NEW folder for pooled, class-organized results.")
    parser.add_argument("--confidence_threshold", type=float, default=0.0,
                         help="If set > 0, photos below this confidence go into an 'Uncertain' folder.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path(args.output)

    if not root.is_dir():
        raise FileNotFoundError(f"--root path does not exist or is not a folder: {root}")
    if root.resolve() == output_root.resolve():
        raise ValueError("--output must be a different folder from --root.")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    tower_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("tower_"))
    if not tower_dirs:
        raise ValueError(f"No tower_* folders found under {root}")

    print(f"Loading trained model from {args.model} ...")
    bundle = joblib.load(args.model)
    classifier = bundle["classifier"]
    class_names = bundle["class_names"]
    print(f"Model classes: {class_names}")

    print("Loading frozen ResNet18 backbone ...")
    backbone = load_backbone(device=args.device)
    transform = build_transform()

    grand_total = {}
    log_rows = []   # tower, original_filename, predicted_class, confidence, pooled_filename

    for tower_dir in tower_dirs:
        rgb_dir = tower_dir / "rgb"
        if not rgb_dir.is_dir():
            print(f"[skip] {tower_dir.name}: no 'rgb' subfolder")
            continue

        image_paths = list_flat_images(rgb_dir)
        if not image_paths:
            print(f"{tower_dir.name}: no images found")
            continue

        print(f"\n{tower_dir.name}: classifying {len(image_paths)} photos ...")
        tower_counts = {}

        for path in image_paths:
            feature = extract_feature(backbone, transform, path, device=args.device).reshape(1, -1)
            probs = classifier.predict_proba(feature)[0]
            pred_idx = probs.argmax()
            pred_class = class_names[pred_idx]
            confidence = float(probs[pred_idx])

            if args.confidence_threshold > 0 and confidence < args.confidence_threshold:
                dest_class = "Uncertain"
            else:
                dest_class = pred_class

            # Pooled, flat storage: output/<class>/<tower>_<original_filename>
            pooled_name = f"{tower_dir.name}_{path.name}"
            dest_dir = output_root / dest_class
            dest_path = dest_dir / pooled_name

            if args.dry_run:
                print(f"  {path.name}  ->  {dest_class}/{pooled_name}  (confidence={confidence:.3f})")
            else:
                dest_dir.mkdir(parents=True, exist_ok=True)
                if dest_path.exists():
                    print(f"  [warning] already exists, skipping: {dest_path}")
                else:
                    shutil.copy2(str(path), str(dest_path))  # always copy -- root stays untouched

            tower_counts[dest_class] = tower_counts.get(dest_class, 0) + 1
            grand_total[dest_class] = grand_total.get(dest_class, 0) + 1
            log_rows.append([tower_dir.name, path.name, pred_class, f"{confidence:.4f}", dest_class, pooled_name])

        print(f"  {tower_dir.name} results: {tower_counts}")

    print(f"\n=== Grand total across all towers ===")
    print(grand_total)

    if not args.dry_run:
        log_path = output_root / "classification_log.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tower", "original_filename", "predicted_class", "confidence", "sorted_into", "pooled_filename"])
            writer.writerows(log_rows)
        print(f"\nResults written to: {output_root.resolve()}")
        print(f"  - Pooled by class: {output_root}/Phase, {output_root}/Top, {output_root}/Earth_wire, etc.")
        print(f"  - Log (needed by sort.py): {log_path}")
        print(f"Input folder untouched: {root.resolve()}")


if __name__ == "__main__":
    main()

import argparse
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
from pathlib import Path

parser = argparse.ArgumentParser(description="Build the (X, y) training pair from a flat class-organized data folder.")
parser.add_argument("--data_root", type=str, required=True,
                     help="Path to the flat class-organized folder (contains Phase/, Top/, Earth_wire/, etc.)")
parser.add_argument("--augment_per_image", type=int, default=4,
                     help="Number of extra augmented views to generate per source image (0 = disable augmentation).")
parser.add_argument("--output_prefix", type=str, default="build",
                     help="Prefix for output files: <prefix>_X.npy, <prefix>_y.npy, <prefix>_paths.txt")
args = parser.parse_args()

AUGMENT_PER_IMAGE = args.augment_per_image

# --- Rebuild the frozen backbone ---
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval()
for param in model.parameters():
    param.requires_grad = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Base (non-augmented) preprocessing -- resize, center crop, normalize.
# Always used once per image, regardless of AUGMENT_PER_IMAGE.
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Augmented preprocessing -- random crop/flip/rotation/color-jitter, then
# the same final size + normalization as base_transform. Used to generate
# EXTRA feature vectors per image (same label, slightly different view),
# synthetically expanding the effective training set -- most useful for
# your smaller classes (Top, Earth_wire) which have very few source photos.
augment_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def extract_feature(image_path, transform):
    """Turns one image file into a 512-dim numpy vector, using whichever
    transform (base or augmented) is passed in."""
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        feature = model(tensor)  # (1, 512)
    return feature.squeeze(0).numpy()  # (512,)


# --- Point this at your flat class-organized data/ folder ---
data_root = Path(args.data_root)

# Only include classes that actually have images -- skips "Bottom" automatically
# for now since it's empty, without you having to remember to exclude it by hand.
class_dirs = sorted(
    d for d in data_root.iterdir()
    if d.is_dir() and any(p.suffix.lower() in IMAGE_EXTENSIONS for p in d.iterdir())
)
class_names = [d.name for d in class_dirs]
print(f"Classes found (with at least 1 image): {class_names}")
print(f"Augmented views per image: {AUGMENT_PER_IMAGE} (plus 1 base view = "
      f"{AUGMENT_PER_IMAGE + 1} total rows per source image)")

all_features = []   # will become X: list of (512,) vectors -> stacked into (N, 512)
all_labels = []      # will become y: list of class-name strings, one per row of X
all_paths = []        # keeps track of which SOURCE file each row came from (repeated
                       # for augmented views of the same image), for debugging later

for class_dir in class_dirs:
    image_paths = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"\nProcessing class '{class_dir.name}': {len(image_paths)} source images "
          f"-> {len(image_paths) * (AUGMENT_PER_IMAGE + 1)} rows after augmentation")

    for i, path in enumerate(image_paths, 1):
        # 1 base (non-augmented) view -- always included.
        feat = extract_feature(path, base_transform)
        all_features.append(feat)
        all_labels.append(class_dir.name)   # <-- this is the label-attaching step
        all_paths.append(path)

        # AUGMENT_PER_IMAGE extra augmented views of the SAME source image,
        # SAME label -- each call to augment_transform applies fresh random
        # crop/flip/rotation/jitter, so every augmented view is different
        # even though they all come from the same original photo.
        for _ in range(AUGMENT_PER_IMAGE):
            aug_feat = extract_feature(path, augment_transform)
            all_features.append(aug_feat)
            all_labels.append(class_dir.name)
            all_paths.append(path)

        if i % 50 == 0 or i == len(image_paths):
            print(f"  [{i}/{len(image_paths)}] source images processed")

# Stack everything into the final training pair.
X = np.stack(all_features)          # shape (N_total, 512)
y = np.array(all_labels)            # shape (N_total,) -- strings like "Phase", "Top", ...

print(f"\n=== Final training pair ===")
print(f"X shape: {X.shape}")   # (N_total, 512)
print(f"y shape: {y.shape}")   # (N_total,)

# Sanity check: count how many rows belong to each class -- this should
# exactly match (source image count) x (AUGMENT_PER_IMAGE + 1) per class.
print("\nClass distribution in y:")
unique_labels, counts = np.unique(y, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"  {label}: {count}")

# Sanity check: confirm row i of X really corresponds to row i of y and
# all_paths, by inspecting a couple of examples.
print("\nSpot check (first row and last row):")
print(f"  Row 0:  label='{y[0]}'   file={all_paths[0].name}")
print(f"  Row -1: label='{y[-1]}'  file={all_paths[-1].name}")

# Extra sanity check specific to augmentation: confirm the base view and an
# augmented view of the SAME image produce DIFFERENT feature vectors (proves
# augmentation is actually doing something), while still being closer to
# each other than to a totally different image (proves it's not so extreme
# that it destroys the image's identity).
if AUGMENT_PER_IMAGE > 0 and len(all_features) >= AUGMENT_PER_IMAGE + 2:
    base_vec = all_features[0]
    aug_vec = all_features[1]  # first augmented view of the same source image as row 0
    other_vec = all_features[AUGMENT_PER_IMAGE + 1]  # base view of the NEXT source image
    diff_same_image = np.abs(base_vec - aug_vec).mean()
    diff_diff_image = np.abs(base_vec - other_vec).mean()
    print(f"\nAugmentation sanity check:")
    print(f"  Base vs. augmented view of the SAME image:  mean abs diff = {diff_same_image:.4f}")
    print(f"  Base view vs. a DIFFERENT image:             mean abs diff = {diff_diff_image:.4f}")
    print(f"  (First number should be > 0 but typically smaller than the second)")

# Save everything so training doesn't need to recompute features from scratch every time.
np.save(f"{args.output_prefix}_X.npy", X)
np.save(f"{args.output_prefix}_y.npy", y)
with open(f"{args.output_prefix}_paths.txt", "w") as f:
    for path in all_paths:
        f.write(f"{path}\n")
print(f"\nSaved {args.output_prefix}_X.npy, {args.output_prefix}_y.npy, {args.output_prefix}_paths.txt")

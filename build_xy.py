import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
from pathlib import Path

# --- Rebuild the frozen backbone ---
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval()
for param in model.parameters():
    param.requires_grad = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def extract_feature(image_path):
    """Turns one image file into a 512-dim numpy vector (same as Steps 2-3)."""
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        feature = model(tensor)  # (1, 512)
    return feature.squeeze(0).numpy()  # (512,)


# --- Point this at your flat class-organized data/ folder ---
data_root = Path("/home/om/Documents/NOTES/Samarth/ML/Image_classification_layer_1/Trained_data")

# Only include classes that actually have images -- skips "Bottom" automatically
# for now since it's empty, without you having to remember to exclude it by hand.
class_dirs = sorted(
    d for d in data_root.iterdir()
    if d.is_dir() and any(p.suffix.lower() in IMAGE_EXTENSIONS for p in d.iterdir())
)
class_names = [d.name for d in class_dirs]
print(f"Classes found (with at least 1 image): {class_names}")

all_features = []   # will become X: list of (512,) vectors -> stacked into (N, 512)
all_labels = []      # will become y: list of class-name strings, one per row of X
all_paths = []        # keeps track of which file each row came from, for debugging later

for class_dir in class_dirs:
    image_paths = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"\nProcessing class '{class_dir.name}': {len(image_paths)} images")

    for i, path in enumerate(image_paths, 1):
        feat = extract_feature(path)
        all_features.append(feat)
        all_labels.append(class_dir.name)   # <-- this is the label-attaching step
        all_paths.append(path)
        if i % 50 == 0 or i == len(image_paths):
            print(f"  [{i}/{len(image_paths)}] processed")

# Stack everything into the final training pair.
X = np.stack(all_features)          # shape (N_total, 512)
y = np.array(all_labels)            # shape (N_total,) -- strings like "Phase", "Top", ...

print(f"\n=== Final training pair ===")
print(f"X shape: {X.shape}")   # (N_total, 512)
print(f"y shape: {y.shape}")   # (N_total,)

# Sanity check: count how many rows belong to each class -- this should
# exactly match the per-class counts printed above.
print("\nClass distribution in y:")
unique_labels, counts = np.unique(y, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"  {label}: {count}")

# Sanity check: confirm row i of X really corresponds to row i of y and
# all_paths, by inspecting a couple of examples.
print("\nSpot check (first row and last row):")
print(f"  Row 0:  label='{y[0]}'   file={all_paths[0].name}")
print(f"  Row -1: label='{y[-1]}'  file={all_paths[-1].name}")

# Save everything  (train/test split + LogisticRegression + k-fold CV)
# doesn't need to recompute features from scratch every time.
np.save("build_X.npy", X)
np.save("build_y.npy", y)
with open("build_paths.txt", "w") as f:
    for path in all_paths:
        f.write(f"{path}\n")
print("\nSaved build_X.npy, build_y.npy, build_paths.txt")

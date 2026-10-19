# Layer 1 Tower Component Classification — Full Script Reference

This covers every script in the pipeline, in the order they run, with what
each one does, its inputs/outputs, and the exact command to run it. Use
this as the spec if you need to recreate any of these scripts from
scratch later.

---

## Pipeline overview

```
Result_data (labeled, per-tower, circuit_1/circuit_2/earthwire/tower_centre)
       │
       ▼  reorganize_tower_folders.py
Result_data_organized (labeled, per-tower, Phase/Top/Earth_wire/Bottom)
       │
       ▼  build_training_dataset.py
data/ (labeled, flat by class — Phase/, Top/, Earth_wire/, Bottom/)
       │
       ▼  RestNet18_testing.py-style feature extraction + build_xy.py
build_X.npy, build_y.npy  (the X, y training pair)
       │
       ▼  train_classifier.py
layer1_model.joblib  (the trained model)
       │
       ▼  classify.py  (on NEW unlabeled towers)
Classified_output/  (pooled flat by predicted class, + classification_log.csv)
       │
       ▼  sort.py
Sorted_classified_output_by_tower/  (per-tower mirrored structure, predicted classes)
```

---

## 1. `reorganize_tower_folders.py`

**Purpose:** Converts the original labeled data (`circuit_1/circuit_2/earthwire/tower_centre`
subfolders per tower) into the simpler 4-class layout (`Phase/Top/Earth_wire/Bottom`)
that the rest of the pipeline expects. Reads from `Result_data`, never modifies
it — writes a brand new folder.

**Class mapping:**
| Output class | Reads from |
|---|---|
| `Phase` | `circuit_1/{B,R,Y}-phase` AND `circuit_2/{B,R,Y}-phase`, pooled |
| `Top` | `tower_centre` |
| `Earth_wire` | `earthwire` |
| `Bottom` | `bottom` (created empty if source doesn't exist) |

**Input:** `Result_data/tower_X/rgb/circuit_1/B-phase/*.JPG` etc. (read-only)
**Output:** `Result_data_organized/tower_X/rgb/{Phase,Top,Earth_wire,Bottom}/*.JPG`

**Command:**
```bash
python reorganize_tower_folders.py --root Result_data --output Result_data_organized --dry_run
python reorganize_tower_folders.py --root Result_data --output Result_data_organized
```

**Key implementation details:**
- `--root` and `--output` must be different paths (script refuses if identical).
- Filenames are preserved exactly — no prefixing (since it's still per-tower here).
- Detects filename collisions within the same tower (e.g. same name in both
  circuit_1 and circuit_2) and reports them instead of silently overwriting.
- Safe to re-run: already-copied files are detected and skipped, not duplicated.
- `--move` relocates instead of copying (only use once copy-mode output is verified).

---

## 2. `build_training_dataset.py`

**Purpose:** Pools the per-tower `Phase/Top/Earth_wire/Bottom` folders across
ALL towers into one flat, class-organized dataset — the shape a training
script expects (one subfolder per class, images from every tower combined).

**Input:** `Result_data_organized/tower_X/rgb/{Phase,Top,Earth_wire,Bottom}/*.JPG`
**Output:** `data/{Phase,Top,Earth_wire,Bottom}/tower_X_filename.JPG`

**Command:**
```bash
python build_training_dataset.py --root Result_data_organized --output data --dry_run
python build_training_dataset.py --root Result_data_organized --output data
```

**Key implementation details:**
- Filenames prefixed with tower name (`tower_1_DJI_..._0034_V.JPG`) to avoid
  collisions when pooling across towers.
- **Disk space safety check** up front: compares total source data size vs.
  free space at destination, warns if too close (common cause of silent
  0-byte/truncated copies).
- **Skips already-0-byte source files** with a clear warning, rather than
  blindly copying corruption forward.
- **Verifies every copy**: checks destination file size matches source size
  exactly after copying; reports any mismatch immediately.
- `--move` available; `--root`/`--output` must differ.

---

## 3. Feature extraction pattern (used by `RestNet18_testing.py`, `build_xy.py`, `classify.py`)

This isn't a single script — it's the core building block reused everywhere.
Documenting it once here since every downstream script depends on it.

**The frozen backbone:**
```python
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
model.fc = torch.nn.Identity()      # strip the 1000-class ImageNet head
model.eval()
for param in model.parameters():
    param.requires_grad = False      # frozen -- never trained on tower photos
```

**Preprocessing (must match ImageNet's original normalization):**
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

**One image -> one 512-dim feature vector:**
```python
def extract_feature(model, transform, image_path, device="cpu"):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)   # (1, 3, 224, 224)
    with torch.no_grad():
        feature = model(tensor)                          # (1, 512)
    return feature.squeeze(0).cpu().numpy()               # (512,)
```

**Sanity checks worth keeping in any script that does this:**
- Pixel value range after normalize should be roughly `[-2.5, 2.5]`, not `[0,1]` or `[0,255]`.
- Same image run twice through the frozen model should give bit-identical output
  (confirms no hidden randomness — dropout/batchnorm — in eval mode).
- Two different images should give a feature vector with a clearly nonzero
  mean absolute difference (confirms the extractor is actually distinguishing images).

---

## 4. `build_xy.py`

**Purpose:** Walks every class folder in `data/`, extracts a feature vector
for every image, and builds the final `(X, y)` training pair — `X` is the
feature matrix, `y` is the matching array of class labels.

**Input:** `data/{Phase,Top,Earth_wire}/*.JPG` (Bottom auto-skipped if empty)
**Output:** `build_X.npy` (shape `(N, 512)`), `build_y.npy` (shape `(N,)`),
`build_paths.txt` (which file each row came from)

**Command:**
```bash
python build_xy.py
```
(No CLI args in the version built here — `data_root` is a hardcoded path at
the top of the script; edit it directly, or add `argparse` if preferred.)

**Key implementation details:**
- Classes are auto-discovered: any subfolder of `data/` containing at least
  one image is included; empty folders (like `Bottom`, before you have that
  data) are automatically skipped rather than crashing.
- **Resilient to corrupted/unreadable files**: wraps `extract_feature()` in a
  try/except; any file that fails (e.g. `PIL.UnidentifiedImageError` from a
  0-byte or corrupted file) is logged to `step4_skipped_files.txt` and
  skipped, rather than crashing the whole run.
- Spot-check printed at the end: confirms row 0 and row -1 of `y` actually
  match the class folder their feature came from — catches label/row
  misalignment bugs early.

---

## 5. `train_classifier.py`

**Purpose:** Trains the actual classifier — multinomial logistic regression
(softmax regression) on top of the frozen ResNet18 features — with
stratified k-fold cross-validation for an honest accuracy estimate, then
trains a final model on 100% of the data.

**Input:** `build_X.npy`, `build_y.npy`
**Output:** `layer1_model.joblib` (contains the trained classifier, label
encoder, and class names — everything `classify.py` needs later)

**Command:**
```bash
python train_classifier.py
```

**Key implementation details:**
- `LabelEncoder` converts string labels (`"Phase"`, `"Top"`, ...) to integers
  for training; the encoder itself is saved in the `.joblib` bundle so
  predictions can be mapped back to readable class names later.
- `StratifiedKFold` (default `K_FOLDS = 5`) — stratification keeps each
  fold's class proportions matching the overall dataset, important given
  class imbalance (`Phase` far outnumbers `Top`/`Earth_wire`).
- `LogisticRegression(class_weight="balanced")` — up-weights minority
  classes during training so the model doesn't just default to predicting
  the majority class (`Phase`) for everything.
- Reports **per-fold accuracy**, **mean CV accuracy ± std**, a full
  **per-class precision/recall/f1 report**, and a **confusion matrix** —
  the per-class report matters more than overall accuracy, since accuracy
  alone can look deceptively good while small classes (`Top`, `Earth_wire`)
  quietly underperform.
- `RUN_CV = True/False` toggle at the top: set `False` to skip the k-fold
  sanity check entirely (e.g. if you already have a separate real test set)
  and go straight to training the final model on all the data.
- **The final saved model is always trained on 100% of the data**, regardless
  of `RUN_CV` — the k-fold loop is purely for evaluation, never for holding
  data back from the deployed model.

**Reading the output:**
- `k_folds=1` is invalid (crashes) — a fold must have something to hold out
  *and* something to train on; there's no such thing as a 1-fold split.
- **Macro avg** = simple average across classes, each class weighted equally
  regardless of size — the more honest number when checking if small classes
  are being learned.
- **Weighted avg** = average weighted by class size (`support`) — dominated
  by whichever class has the most images (`Phase`), so it can look great
  even if small classes are failing.
- **Confusion matrix** rows = true class, columns = predicted class; the
  diagonal is correct predictions, everything off-diagonal shows exactly
  which classes get confused with which.

---

## 6. `classify.py`

**Purpose:** Runs the trained model on brand-new, UNLABELED tower photos.
Pools results flat by predicted class across all towers (not per-tower).
`--root` is read-only.

**Input:** `Input_data/tower_X/rgb/*.JPG` (flat, unlabeled)
**Output:** `Classified_output/{Phase,Top,Earth_wire}/tower_X_filename.JPG`,
plus `Classified_output/classification_log.csv`

**Command:**
```bash
python classify.py --root Input_data --model layer1_model.joblib --output Classified_output --dry_run
python classify.py --root Input_data --model layer1_model.joblib --output Classified_output
```

**Key implementation details:**
- For every photo: extracts its 512-dim feature (same frozen backbone as
  training), gets `predict_proba()` from the classifier, and takes the
  highest-probability class as the prediction — that highest probability
  IS the "confidence" score.
- Filenames prefixed with tower name when pooled (`tower_22_DJI_...JPG`) to
  avoid collisions across towers.
- `--confidence_threshold X`: predictions below `X` get routed to an
  `Uncertain/` folder instead of being trusted outright — useful since this
  data has no ground truth to check against.
- `classification_log.csv` columns: `tower, original_filename,
  predicted_class, confidence, sorted_into, pooled_filename` — this file is
  what `sort.py` reads to reconstruct the per-tower view without re-running
  the model.
- `--root` and `--output` must differ; input is always copied, never moved.

---

## 7. `sort.py`

**Purpose:** Takes `classify.py`'s pooled, flat-by-class output and
reorganizes it into a per-tower mirrored structure — WITHOUT re-running the
model (fast; just reads the CSV log and copies files).

**Input:** `Classified_output/` (must contain `classification_log.csv`)
**Output:** `Sorted_classified_output_by_tower/tower_X/rgb/{Phase,Top,Earth_wire}/original_filename.JPG`,
plus a `predictions_log.csv` slice per tower

**Command:**
```bash
python sort.py --input Classified_output --output Sorted_classified_output_by_tower --dry_run
python sort.py --input Classified_output --output Sorted_classified_output_by_tower
```

**Key implementation details:**
- No model, no ResNet, no `torch` import at all — purely reads
  `classification_log.csv` and copies files based on it.
- Restores each photo's **original filename** (strips the tower prefix that
  `classify.py` added when pooling).
- Reports any log rows whose source file is missing from `--input` (e.g. if
  the pooled folder was modified or incomplete) instead of crashing.
- `--input` and `--output` must differ; always copies, never moves.

---

## Design principles used throughout this pipeline

These are worth carrying into any new script you write later:

1. **`--root`/`--input` is read-only; `--output` is always a separate,
   brand-new folder.** Every script refuses to run if they're the same path.
   Nothing that touches real data ever modifies it in place.
2. **`--dry_run` first, always.** Every script supports printing its full
   plan without touching a single file, so you can sanity-check before
   committing.
3. **Copy by default, `--move` opt-in.** Destructive operations require an
   explicit flag.
4. **Idempotent re-runs.** If a script is interrupted or run twice, it
   detects already-existing output and skips (not duplicate, not crash).
5. **Fail loud on data problems, not on code problems.** 0-byte files,
   corrupted images, size mismatches after copying — all get logged clearly
   and skipped, rather than crashing the whole run or silently propagating
   bad data forward.
6. **Every copy step can be verified.** Either by size-matching (during
   copy) or by a `diff` of before/after file listings (as done manually
   during testing) — never assume a copy succeeded silently.

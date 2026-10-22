import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# --- Load the X, y pair saved by Step 4 ---
X = np.load("build_X.npy")
y = np.load("build_y.npy")
with open("build_paths.txt", encoding="utf-8") as f:
    groups = np.array([line.rstrip("\r\n") for line in f])

print(f"Loaded X: {X.shape}")
print(f"Loaded y: {y.shape}")

if len(X) != len(y) or len(X) != len(groups):
    raise ValueError(
        "build_X.npy, build_y.npy, and build_paths.txt must have the same "
        f"number of rows; got {len(X)}, {len(y)}, and {len(groups)}."
    )

unique_labels, counts = np.unique(y, return_counts=True)
print("\nClass distribution (full dataset):")
for label, count in zip(unique_labels, counts):
    print(f"  {label}: {count}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# --- Make a stratified split while keeping each source image in one fold ---
# Each path appears once for the base image and again for its augmented views.
# Grouping by path prevents versions of the same original image from appearing
# in both training and test data.
K_FOLDS = 5
skf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Use the first of the 5 stratified group folds as the held-out test set.
train_idx, test_idx = next(skf.split(X, y_encoded, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

print(f"\nTrain set size: {len(train_idx)} feature rows, "
      f"{len(np.unique(groups[train_idx]))} source images")
print(f"Test set size:  {len(test_idx)} feature rows, "
      f"{len(np.unique(groups[test_idx]))} source images")

print("\nClass distribution in TRAIN set:")
train_labels, train_counts = np.unique(encoder.inverse_transform(y_train), return_counts=True)
for label, count in zip(train_labels, train_counts):
    print(f"  {label}: {count}")

print("\nClass distribution in TEST set:")
test_labels, test_counts = np.unique(encoder.inverse_transform(y_test), return_counts=True)
for label, count in zip(test_labels, test_counts):
    print(f"  {label}: {count}")

# --- Train ONLY on the training portion (folds 2-5) ---
clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", solver="lbfgs")
clf.fit(X_train, y_train)

# --- Evaluate on the held-out portion (fold 1) -- this is your real test result ---
y_pred = clf.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"\nTest accuracy (on held-out fold): {accuracy:.3f}")

report_path = Path("test_predictions.csv")
with report_path.open("w", newline="", encoding="utf-8-sig") as report_file:
    writer = csv.DictWriter(
        report_file,
        fieldnames=[
            "feature_row",
            "image_path",
            "image_exists",
            "true_label",
            "predicted_label",
            "correct",
        ],
    )
    writer.writeheader()
    for row_index, actual, predicted in zip(test_idx, y_test, y_pred):
        image_path = Path(groups[row_index]).expanduser().resolve()
        writer.writerow({
            "feature_row": int(row_index),
            "image_path": str(image_path),
            "image_exists": image_path.is_file(),
            "true_label": encoder.inverse_transform([actual])[0],
            "predicted_label": encoder.inverse_transform([predicted])[0],
            "correct": bool(actual == predicted),
        })
print(f"Saved held-out test predictions to {report_path.resolve()}")

print("\nPer-class report (evaluated on the held-out fold only):")
print(classification_report(y_test, y_pred, target_names=list(encoder.classes_), zero_division=0))

print("Confusion matrix (rows = true class, columns = predicted class):")
print("Classes in order:", list(encoder.classes_))
print(confusion_matrix(y_test, y_pred))

# --- Save this model (trained on the 4-fold training portion only) ---
# NOTE: unlike step5_train_classifier.py, this model was trained on only
# ~80% of your data (folds 2-5), not the full dataset -- since you're using
# the remaining fold purely to test. If you want a model trained on
# everything for actual deployment, retrain on the full X, y afterward
# (see step5_train_classifier.py's "final_clf" step).
joblib.dump({
    "classifier": clf,
    "label_encoder": encoder,
    "class_names": list(encoder.classes_),
}, "layer1_model.joblib")
print("\nSaved model (trained on folds 2-5 only) to step5b_single_split_model.joblib")

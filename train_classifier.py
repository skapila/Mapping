import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# --- Load the X, y pair saved by Step 4 ---
X = np.load("build_X.npy")
y = np.load("build_y.npy")

print(f"Loaded X: {X.shape}")
print(f"Loaded y: {y.shape}")

unique_labels, counts = np.unique(y, return_counts=True)
print("\nClass distribution (full dataset):")
for label, count in zip(unique_labels, counts):
    print(f"  {label}: {count}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# --- Build the SAME 5 folds as before, but only use ONE of them as the test set ---
# We reuse StratifiedKFold (not train_test_split) specifically so this is
# literally "fold 5" in the same sense as the 5-fold CV script -- same
# random_state, same fold boundaries -- just using 1 split instead of
# rotating through all 5.
K_FOLDS = 5
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# .split() is a generator that yields 5 (train_idx, test_idx) pairs, one per
# fold. We only want the first one -- "fold 1" as the test set, folds 2-5
# combined as training. (If you specifically want fold 5 as test instead,
# see the note below the code.)
all_splits = list(skf.split(X, y_encoded))
train_idx, test_idx = all_splits[0]   # <-- index 0 = "fold 1" as test set

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

print(f"\nTrain set size: {len(train_idx)} images  (folds 2-5 combined)")
print(f"Test set size:  {len(test_idx)} images  (fold 1 only)")

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

"""
=========================================================
 DecodeLabs | Project 2 - Data Classification Using AI
 Track: Supervised Learning | Algorithm: K-Nearest Neighbors
=========================================================

Pipeline (as per the IPO Framework in the training deck):

  INPUT   -> Load Iris dataset, explore it, scale features (StandardScaler)
  PROCESS -> Train-Test split, tune K (elbow method), train KNN model
  OUTPUT  -> Confusion Matrix, F1 Score, Accuracy, Classification Report

Goal: Build a basic classification model using a small dataset
      (Iris: 150 samples, 3 classes, 4 features) and prove you can
      train, test, and validate an AI model end-to-end.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    f1_score,
)

# ---------------------------------------------------------
# STEP 1 (INPUT): Load and understand the dataset
# ---------------------------------------------------------
print("=" * 60)
print("STEP 1: LOAD & UNDERSTAND THE DATASET (Iris Benchmark)")
print("=" * 60)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(f"\nSamples: {df.shape[0]}  |  Features: {iris.data.shape[1]}  |  Classes: {len(iris.target_names)}")
print(f"Class names: {list(iris.target_names)}")
print("\nFirst 5 rows:\n", df.head())
print("\nClass distribution (balanced dataset check):\n", df["species"].value_counts())
print("\nStatistical summary:\n", df.describe())

# Separate features (X) and target labels (y)
X = iris.data
y = iris.target

# ---------------------------------------------------------
# STEP 2 (INPUT): Feature scaling - "The Gatekeeper Rule"
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: FEATURE SCALING (StandardScaler: mean=0, var=1)")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nRaw sample (row 0):   ", np.round(X[0], 2))
print("Scaled sample (row 0):", np.round(X_scaled[0], 2))
print("\nKNN is distance-based, so scaling prevents features with larger")
print("numeric ranges (e.g. petal length in cm) from dominating the distance calc.")

# ---------------------------------------------------------
# STEP 3 (PROCESS): Train-Test split - "Structural Integrity"
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: TRAIN-TEST SPLIT (80/20, shuffled, stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.20,     # 80% train / 20% test as shown in "The Full Architecture"
    random_state=42,    # reproducibility
    stratify=y,         # keep class balance equal in both sets
    shuffle=True,        # "Randomize before splitting to remove order bias"
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size:     {X_test.shape[0]} samples")

# ---------------------------------------------------------
# STEP 4 (PROCESS): Tune K - "The Elbow Method"
# ---------------------------------------------------------
# NOTE: K is chosen using 5-fold cross-validation on the TRAINING set
# only (never touching X_test). Picking K by scoring against the test
# set would be data leakage and would bias the final evaluation.
print("\n" + "=" * 60)
print("STEP 4: TUNING THE ENGINE - CHOOSING OPTIMAL K")
print("=" * 60)

error_rates = []
k_range = range(1, 26)

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_temp, X_train, y_train, cv=5, scoring="accuracy")
    error_rates.append(1 - scores.mean())

optimal_k = k_range[np.argmin(error_rates)]
print(f"\nOptimal K found at the 'elbow' (via 5-fold CV on training data): K = {optimal_k}")
print("(K is tuned only on training data -- never on the test set -- to avoid leakage.)")

plt.figure(figsize=(8, 5))
plt.plot(k_range, error_rates, marker="o", linestyle="--", color="steelblue")
plt.axvline(optimal_k, color="orangered", linestyle=":", label=f"Optimal K = {optimal_k}")
plt.title("Choosing K: Error Rate vs. K Value")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("elbow_curve.png", dpi=150)
plt.close()
print("Saved plot -> elbow_curve.png")

# ---------------------------------------------------------
# STEP 5 (PROCESS): Instantiate, Fit, Predict - scikit-learn workflow
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: TRAIN THE FINAL KNN MODEL")
print("=" * 60)

model = KNeighborsClassifier(n_neighbors=optimal_k)  # INSTANTIATE
model.fit(X_train, y_train)                          # FIT (memorize the map)
predictions = model.predict(X_test)                  # PREDICT (apply logic)

print(f"\nModel trained with K = {optimal_k} neighbors.")

# ---------------------------------------------------------
# STEP 6 (OUTPUT): Validation - Confusion Matrix, F1, Accuracy
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: OUTPUT VALIDATION (don't trust accuracy alone!)")
print("=" * 60)

acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")

print(f"\nAccuracy: {acc:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print("\nFull Classification Report:\n")
print(classification_report(y_test, predictions, target_names=iris.target_names))

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix (rows=actual, cols=predicted):\n", cm)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix (KNN, K={optimal_k})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("\nSaved plot -> confusion_matrix.png")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE: Input -> Process -> Output")
print("=" * 60)

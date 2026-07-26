# Project 2 — Data Classification Using AI
**DecodeLabs Industrial Training Kit | Batch 2026**

## Goal
Build a basic classification model using a small dataset (Iris), proving mastery
of the supervised learning pipeline: load data → split → train → predict → validate.

## Dataset
**Iris Benchmark** — 150 samples, 3 balanced classes (setosa, versicolor, virginica),
4 features (sepal length/width, petal length/width).

## Pipeline (IPO Framework, per the training deck)

| Stage | Steps |
|---|---|
| **Input** | Load Iris dataset → explore/describe it → scale features with `StandardScaler` |
| **Process** | 80/20 train-test split (shuffled + stratified) → tune K via 5-fold cross-validation on the training set → train `KNeighborsClassifier` |
| **Output** | Confusion Matrix, Accuracy, F1 Score (weighted), full classification report |

## Why these choices
- **StandardScaler**: KNN is distance-based, so unscaled features (e.g. cm-scale
  petal length vs. smaller sepal width) would unfairly dominate the distance
  calculation. Scaling gives every feature mean=0, variance=1.
- **Stratified split**: keeps the 50/50/50 class balance in both train and test sets.
- **K chosen by cross-validation on the training set only** — not on the test set.
  Scoring K against the test set would leak test information into model
  selection and inflate the reported accuracy. This also naturally avoids
  the K=1 "noise/overfitting" trap flagged in the deck's elbow chart.
- **F1 Score over raw accuracy**: the deck's "Accuracy Mirage" slide warns that
  accuracy alone can hide misclassifications, especially in imbalanced settings.
  F1 balances precision and recall.

## Results (this run, `random_state=42`)
- Optimal K: **6** (found via cross-validation, see `elbow_curve.png`)
- Accuracy: **~0.93**
- F1 Score (weighted): **~0.93**
- Confusion Matrix: see `confusion_matrix.png` — the model confuses a couple of
  `virginica` samples with `versicolor`, which is expected since those two
  species overlap in feature space (visible in the deck's decision-boundary slide).

## Files
- `iris_classification.py` — full runnable pipeline (run with `python3 iris_classification.py`)
- `elbow_curve.png` — error rate vs. K, used to pick the optimal K
- `confusion_matrix.png` — final model's confusion matrix on the test set

## How to run
```bash
pip install scikit-learn pandas matplotlib seaborn
python3 iris_classification.py
```

## Possible extensions (per the deck's Conclusion slide: "experiment with unique solutions")
- Compare KNN against another algorithm (e.g. Logistic Regression, Decision Tree)
  on the same split and compare F1 scores.
- Try predicting on a few hand-crafted "new" flower measurements not in the dataset.
- Plot decision boundaries using two features at a time (as shown in the
  "Logic Skeleton" slide).

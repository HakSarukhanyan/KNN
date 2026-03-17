# Breast Cancer Classification with K-Nearest Neighbors (KNN)

A machine learning project that classifies breast tumors as **Malignant (M)** or **Benign (B)** using the K-Nearest Neighbors algorithm, applied to the Wisconsin Breast Cancer dataset.

## Dataset

**File:** `Cancer_Data.csv`

The dataset contains 569 samples with 30 numeric features computed from digitized images of fine needle aspirate (FNA) of breast masses. Each feature describes characteristics of the cell nuclei present in the image.

| Property | Value |
|----------|-------|
| Samples | 569 |
| Features | 30  |
| Target | `diagnosis` — M (malignant) or B (benign) |
| Missing values | None (one fully-null column `Unnamed: 32` is dropped) |

**Key features include:** radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension — each reported as mean, standard error, and worst (largest) value.

## Pipeline Overview

### 1. Exploratory Data Analysis

- Inspected data shape, types, and summary statistics.
- Computed pairwise Spearman correlations across all numeric features.

### 2. Feature Engineering

- **Dropped** the `id` column and the empty `Unnamed: 32` column.
- **Removed highly correlated features** (|r| ≥ 0.8) to reduce dimension and multicollinearity, bringing the feature set down from 30 to 13 features.

### 3. Outlier Removal

- Reviewed distributions of each remaining feature using histograms.
- Manually removed the single most extreme outlier from 9 features (`texture_mean`, `smoothness_mean`, `radius_se`, `texture_se`, `smoothness_se`, `compactness_se`, `concave points_se`, `symmetry_se`, `symmetry_worst`).


### 4. Preprocessing

- **Scaling:** MinMaxScaler applied to normalize features to the [0, 1] range.
- **Dimensionality reduction:** PCA with 6 components, capturing approximately 89% of total variance.

  | Component | Variance Explained |
  |-----------|-------------------|
  | PC1 | 36.9% |
  | PC2 | 17.7% |
  | PC3 | 13.3% |
  | PC4 | 8.2% |
  | PC5 | 7.2% |
  | PC6 | 5.9% |

### 5. Model Selection via Cross-Validation

- **5-Fold Stratified Cross-Validation** tested KNN with k = 1 through 20.
- Scaling and PCA were fitted within each fold to prevent data leakage.
- Best cross-validation accuracy: **93.9%** at k = 13.
- Observed similar performance across odd values of k in the range [5, 19].

### 6. Final Evaluation

- **Train/Test split:** 80/20, stratified by diagnosis, random state 32.
- Evaluated all odd k from 5 to 19 on the held-out test set.
- **Best test performance at k = 5** with accuracy of **96.4%**.

## Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 96.4% |
| **Precision** | 1.000 |
| **Recall** | 0.902 |
| **F1 Score** | 0.949 |

**Confusion Matrix** (test set, n = 112):

|  | Predicted B | Predicted M |
|--|-------------|-------------|
| **Actual B** | 71 | 0 |
| **Actual M** | 4 | 37 |

The model achieves perfect precision (no false positives) — every tumor predicted as malignant is truly malignant. The 4 false negatives (malignant tumors classified as benign) represent the primary area for improvement.

## Requirements


- **pandas** — 2.3.3
- **NumPy** — 2.2.6
- **scikit-learn** — 1.7.2 
- **Matplotlib** — 3.10.7

## Project Structure

```
├── Cancer_Data.csv    #  dataset
├── knn.ipynb          # Full analysis notebook (EDA → model → evaluation)
└── README.md
```

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```
2. Place `Cancer_Data.csv` in the same directory as the notebook.
3. Open and run `knn.ipynb` sequentially.

## Possible Improvements

- Use a more principled approach for dropping correlated features (e.g., VIF or feature importance ranking) instead of dropping the second column in each pair.
- Apply IQR-based or statistical outlier detection rather than manual removal.
- Experiment with weighted KNN or other distance metrics (Manhattan, Mahalanobis).

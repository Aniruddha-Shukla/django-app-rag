# Machine Learning Salary Predictor: Random Forest vs XGBoost

Week 2 of my **Learning in Public** AI Career Copilot roadmap. This project builds an end-to-end regression pipeline that predicts Data Analyst salaries from job posting features — technical skills mentioned in the description and job location.

## Model Architecture

Tree-based ensemble methods are a strong fit for tabular data like job postings:

- **Random Forest** averages predictions across many decorrelated decision trees, which reduces overfitting and handles mixed feature types (binary skill flags + one-hot location columns) without manual scaling.
- **XGBoost** builds trees sequentially, with each new tree correcting the residual errors of the ensemble. It often achieves strong performance on structured datasets and exposes `feature_importances_` for interpretability.

Both models were trained on the same feature matrix with an 80/20 train/test split (`random_state=42`).

### Features

| Category | Features |
|---|---|
| **Skills** (binary) | `python`, `sql`, `excel`, `tableau`, `aws`, `spark` — detected in the Job Description |
| **Location** (one-hot) | Top 10 most frequent cities (e.g., San Francisco, New York, Chicago) |
| **Target** | `Avg_Salary` — numeric average parsed from the salary range string |

## Results

| Model | MAE | R² |
|---|---|---|
| Random Forest | $17,292 | 0.097 |
| XGBoost | $17,798 | 0.051 |

Both models explain a modest share of salary variance, which is expected: real-world compensation depends on experience, company size, and seniority — fields not fully captured here.

## Key Findings

The XGBoost feature importance chart (`feature_importance.png`) reveals a clear pattern:

**Location dominates salary prediction.** The top 8 features are all cities, not skills. San Francisco (`loc_San Francisco, CA`) is the single strongest predictor (importance ≈ 0.32), followed by Dallas, San Diego, and Austin.

**Skills matter, but less than geography.** Only `spark` (0.018) and `aws` (0.011) appear in the top 10 — and both rank well below any location feature. Python, SQL, Excel, and Tableau did not break into the top 10, suggesting that *where* you work currently carries more predictive signal than *which* tools appear in the posting text for this dataset.

**Practical takeaway:** For Data Analyst roles in this sample, targeting high-cost markets (especially the Bay Area) is associated with higher predicted salaries, while listing cloud/big-data skills (`aws`, `spark`) provides a smaller but measurable boost.

## Getting Started

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **macOS note:** If XGBoost fails to load, install OpenMP first:
> ```bash
> brew install libomp
> ```

### 2. Add your data

Place `DataAnalyst.csv` in the project root (same directory as `train_model.py`). The file is gitignored to keep large datasets off GitHub.

### 3. Train and evaluate

```bash
python train_model.py
```

This will:
- Load and clean salary data
- Engineer skill and location features
- Train Random Forest and XGBoost models
- Print MAE and R² for both models
- Save `feature_importance.png` to the project root

## Project Structure

```
.
├── DataAnalyst.csv          # Job posting dataset (gitignored)
├── train_model.py           # End-to-end training pipeline
├── feature_importance.png   # XGBoost feature importance chart
├── requirements.txt
├── README.md
└── .gitignore
```

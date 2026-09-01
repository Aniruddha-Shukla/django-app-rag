"""
Week 2 Portfolio Project: Machine Learning Salary Predictor
Trains Random Forest and XGBoost regressors on Data Analyst job postings.
"""

import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = "DataAnalyst.csv"
SKILLS = ["python", "sql", "excel", "tableau", "aws", "spark"]
TOP_N_LOCATIONS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42


def clean_salary(salary_str: str) -> float | None:
    """
    Extract numeric salary from strings like '$37K-$66K (Glassdoor est.)'.

    Parses all dollar amounts, converts K/M suffixes to full values,
    and returns the average of min and max when a range is present.
    Returns None if no valid salary can be parsed.
    """
    if pd.isna(salary_str):
        return None

    # Match patterns such as $37K, $66K, $120M, etc.
    matches = re.findall(r"\$(\d+(?:\.\d+)?)\s*([KkMm])?", str(salary_str))
    if not matches:
        return None

    values = []
    for amount, suffix in matches:
        value = float(amount)
        if suffix and suffix.upper() == "K":
            value *= 1_000
        elif suffix and suffix.upper() == "M":
            value *= 1_000_000
        values.append(value)

    return sum(values) / len(values)


def add_skill_features(df: pd.DataFrame, skills: list[str]) -> pd.DataFrame:
    """Create binary (0/1) columns indicating skill presence in Job Description."""
    descriptions = df["Job Description"].fillna("").str.lower()
    for skill in skills:
        df[f"skill_{skill}"] = descriptions.str.contains(skill, regex=False).astype(int)
    return df


def add_location_features(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, list[str]]:
    """
    One-hot encode the top N most frequent locations.
    Rows with other locations get all-zero location columns.
    """
    top_locations = df["Location"].value_counts().head(top_n).index.tolist()
    df["Location"] = df["Location"].where(df["Location"].isin(top_locations), other="Other")

    location_dummies = pd.get_dummies(df["Location"], prefix="loc", drop_first=False)
    # Drop the catch-all 'Other' column to reduce redundancy
    if "loc_Other" in location_dummies.columns:
        location_dummies = location_dummies.drop(columns=["loc_Other"])

    df = pd.concat([df, location_dummies], axis=1)
    location_cols = location_dummies.columns.tolist()
    return df, location_cols


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Generate predictions and print MAE and R² for a fitted model."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print(f"  Mean Absolute Error (MAE): ${mae:,.0f}")
    print(f"  R-squared (R²):            {r2:.4f}")

    return {"mae": mae, "r2": r2, "predictions": predictions}


def plot_feature_importance(model, feature_names: list[str], top_n: int = 10) -> None:
    """Plot and save a horizontal bar chart of the top N XGBoost feature importances."""
    importances = model.feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        importance_df["feature"],
        importance_df["importance"],
        color="#2563eb",
        edgecolor="#1e40af",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Top 10 XGBoost Feature Importances", fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Annotate bars with importance values
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved feature importance chart to feature_importance.png")


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading DataAnalyst.csv...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df):,} rows")

    # ------------------------------------------------------------------
    # 2. Clean target variable (salary)
    # ------------------------------------------------------------------
    df["Avg_Salary"] = df["Salary Estimate"].apply(clean_salary)
    before_drop = len(df)
    df = df.dropna(subset=["Avg_Salary"])
    print(f"  Dropped {before_drop - len(df):,} rows with missing/invalid salaries")
    print(f"  Remaining rows: {len(df):,}")

    # ------------------------------------------------------------------
    # 3. Engineer skill features
    # ------------------------------------------------------------------
    df = add_skill_features(df, SKILLS)

    # ------------------------------------------------------------------
    # 4. Engineer location features (top 10 cities)
    # ------------------------------------------------------------------
    df, location_cols = add_location_features(df, TOP_N_LOCATIONS)
    skill_cols = [f"skill_{s}" for s in SKILLS]
    feature_cols = skill_cols + location_cols

    # ------------------------------------------------------------------
    # 5. Define X and y, then split 80/20
    # ------------------------------------------------------------------
    X = df[feature_cols]
    y = df["Avg_Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nTrain set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # ------------------------------------------------------------------
    # 6. Train Random Forest
    # ------------------------------------------------------------------
    print("\nTraining RandomForestRegressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    evaluate_model("Random Forest", rf_model, X_test, y_test)

    # ------------------------------------------------------------------
    # 7. Train XGBoost
    # ------------------------------------------------------------------
    print("\nTraining XGBRegressor...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    evaluate_model("XGBoost", xgb_model, X_test, y_test)

    # ------------------------------------------------------------------
    # 8. Feature importance plot (XGBoost)
    # ------------------------------------------------------------------
    plot_feature_importance(xgb_model, feature_cols)


if __name__ == "__main__":
    main()

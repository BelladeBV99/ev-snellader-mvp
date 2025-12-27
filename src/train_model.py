import numpy as np
import pandas as pd
from pathlib import Path

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from joblib import dump


# =========================
# Config
# =========================

DATA_PATH = "data/mvp_train_enriched_500m_comp.csv"
MODEL_PATH = "models/final_ev_sessions_model.cbm"
META_PATH = "models/final_ev_sessions_meta.joblib"

FEATURES = [
    "evse_latitude",
    "evse_longitude",
    "Pool_SiteType",
    "population_1km",
    "osm_office_any_500m",
    "osm_shop_any_500m",
    "competitors_fast_500m",
]

TARGET = "sessions_per_day"
CAT_FEATURES = ["Pool_SiteType"]

RANDOM_SEED = 42


# =========================
# Helpers
# =========================

def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns and clean types."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset mist kolommen: {missing}")

    df = df.copy()

    # target
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if df[TARGET].isna().any():
        raise ValueError("Target sessions_per_day bevat NaN-waarden")

    # numerieke features
    for col in FEATURES:
        if col == "Pool_SiteType":
            df[col] = df[col].fillna("Unknown").astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if len(df) < 30:
        raise ValueError("Te weinig data om een betrouwbaar model te trainen")

    return df


def make_model(verbose=False):
    return CatBoostRegressor(
        iterations=1200,
        depth=6,
        learning_rate=0.08,
        loss_function="MAE",
        random_seed=RANDOM_SEED,
        verbose=verbose,
        allow_writing_files=False,
    )


def make_sample_weights(y: pd.Series, low_q=0.10, high_q=0.90, outlier_weight=0.5):
    """
    Geef outliers (onder p10 en boven p90) minder gewicht tijdens training.
    """
    q_low = float(y.quantile(low_q))
    q_high = float(y.quantile(high_q))

    w = np.ones(len(y), dtype=float)
    w[y < q_low] = outlier_weight
    w[y > q_high] = outlier_weight

    print(
        f"Outlier thresholds: p{int(low_q*100)}={q_low:.2f}, "
        f"p{int(high_q*100)}={q_high:.2f} | outlier_weight={outlier_weight}"
    )
    return w, q_low, q_high


def cross_val_mae_weighted(df: pd.DataFrame, n_splits=5, low_q=0.10, high_q=0.90, outlier_weight=0.5):
    X = df[FEATURES]
    y = df[TARGET].values

    w_all, q_low, q_high = make_sample_weights(df[TARGET], low_q=low_q, high_q=high_q, outlier_weight=outlier_weight)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    maes = []

    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        model = make_model(verbose=False)
        model.fit(
            X.iloc[tr],
            y[tr],
            sample_weight=w_all[tr],
            cat_features=CAT_FEATURES,
        )
        preds = model.predict(X.iloc[te])
        mae = mean_absolute_error(y[te], preds)
        maes.append(mae)
        print(f"Fold {fold}: MAE = {mae:.3f}")

    return float(np.mean(maes)), maes, {"low_q": low_q, "high_q": high_q, "outlier_weight": outlier_weight, "q_low": q_low, "q_high": q_high}


# =========================
# Main
# =========================

def main():
    print("📥 Data laden...")
    df = pd.read_csv(DATA_PATH)
    df = validate_and_clean(df)

    # Hyperparameters for weighting
    LOW_Q = 0.10
    HIGH_Q = 0.90
    OUTLIER_WEIGHT = 0.5

    print("\n🔁 5-fold cross-validation (weighted training, leidend):")
    cv_mae, cv_folds, w_info = cross_val_mae_weighted(
        df,
        n_splits=5,
        low_q=LOW_Q,
        high_q=HIGH_Q,
        outlier_weight=OUTLIER_WEIGHT
    )
    print(f"\n✅ Gemiddelde 5-fold MAE (weighted): {cv_mae:.3f}")

    print("\n🧪 Hold-out evaluatie (sanity check, weighted training):")
    X = df[FEATURES]
    y = df[TARGET]
    weights, q_low, q_high = make_sample_weights(df[TARGET], low_q=LOW_Q, high_q=HIGH_Q, outlier_weight=OUTLIER_WEIGHT)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=RANDOM_SEED
    )

    model = make_model(verbose=100)
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        cat_features=CAT_FEATURES
    )

    preds = model.predict(X_test)
    holdout_mae = mean_absolute_error(y_test, preds)
    print(f"\n📊 Hold-out MAE (weighted): {holdout_mae:.3f}")

    # opslaan
    Path("models").mkdir(exist_ok=True)

    model.save_model(MODEL_PATH)
    dump(
        {
            "features": FEATURES,
            "cat_features": CAT_FEATURES,
            "cv_mae": cv_mae,
            "cv_folds": cv_folds,
            "holdout_mae": holdout_mae,
            "weighting": w_info,
        },
        META_PATH,
    )

    print("\n💾 Model opgeslagen:", MODEL_PATH)
    print("💾 Metadata opgeslagen:", META_PATH)


if __name__ == "__main__":
    main()

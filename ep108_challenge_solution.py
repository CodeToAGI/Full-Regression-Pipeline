"""
EP108 Challenge Solution
Beat R² 0.85 on California Housing
Same pipeline style as the video: clean → engineer → ColumnTransformer → model
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

# Target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# ──────────────────────────────────────────────
# 2. Train/Test split FIRST (no leakage)
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ──────────────────────────────────────────────
# 3. Feature Engineering (train-only logic)
# ──────────────────────────────────────────────
def engineer_features(X):
    X = X.copy()
    # Simple but strong interactions for California Housing
    X["RoomsPerHousehold"] = X["AveRooms"] / X["AveOccup"]
    X["BedroomsPerRoom"]   = X["AveBedrms"] / X["AveRooms"]
    X["PopulationPerHousehold"] = X["Population"] / X["AveOccup"]
    return X

X_train = engineer_features(X_train)
X_test  = engineer_features(X_test)

# ──────────────────────────────────────────────
# 4. Pipeline (identical structure to video)
# ──────────────────────────────────────────────
numeric_features = X_train.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features)
    ]
)

# Three candidate models
models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
}

# ──────────────────────────────────────────────
# 5. Train + Evaluate
# ──────────────────────────────────────────────
results = {}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # 5-fold CV on train set only
    cv_rmse = -cross_val_score(
        pipe, X_train, y_train,
        scoring="neg_root_mean_squared_error",
        cv=5, n_jobs=-1
    )
    
    # Fit on full train
    pipe.fit(X_train, y_train)
    
    # Test predictions
    y_pred = pipe.predict(X_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    
    results[name] = {
        "CV_RMSE_mean": cv_rmse.mean(),
        "CV_RMSE_std":  cv_rmse.std(),
        "Test_RMSE": rmse,
        "Test_MAE":  mae,
        "Test_R2":   r2
    }
    
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"CV RMSE  : {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE : {mae:.4f}")
    print(f"Test R²  : {r2:.4f}")

# Best model summary
best_name = max(results, key=lambda k: results[k]["Test_R2"])
print(f"\n🏆 Best model: {best_name}  →  R² = {results[best_name]['Test_R2']:.4f}")

# ──────────────────────────────────────────────
# 6. Save best pipeline
# ──────────────────────────────────────────────
best_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", models[best_name])
])
best_pipe.fit(X_train, y_train)

joblib.dump(best_pipe, f"california_housing_{best_name.lower()}_r2_{results[best_name]['Test_R2']:.3f}.pkl")
print("\nPipeline saved.")

"""
==============================================================
 Boston House Price Prediction — Full ML Pipeline
==============================================================
 Dataset : HousingData.csv  (506 records, 13 features)
 Target  : MEDV (Median value of owner-occupied homes in $1000s)
 Steps   : EDA -> Preprocessing -> Model Training -> Evaluation
==============================================================
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # headless – no display window needed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# ── 1. Load Dataset ───────────────────────────────────────────────────────────
print("=" * 65)
print("  BOSTON HOUSE PRICE PREDICTION")
print("=" * 65)

df = pd.read_csv("HousingData.csv")
print(f"\n[OK]  Dataset loaded  ->  {df.shape[0]} rows × {df.shape[1]} columns")
print("\n--- First 5 rows ---")
print(df.head())

# ── 2. Exploratory Data Analysis ──────────────────────────────────────────────
print("\n\n--- Data Types & Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
info_df = pd.DataFrame({
    "Dtype": df.dtypes,
    "Missing": missing,
    "Missing %": missing_pct
})
print(info_df)

print("\n--- Descriptive Statistics ---")
print(df.describe().round(2))

# ── 3. Visualisations ─────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=0.9)

# 3a. Distribution of target variable
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["MEDV"].dropna(), kde=True, ax=axes[0], color="#5C7CFA")
axes[0].set_title("Distribution of MEDV (House Price)", fontweight="bold")
axes[0].set_xlabel("MEDV ($1000s)")

from scipy import stats
stats.probplot(df["MEDV"].dropna(), plot=axes[1])
axes[1].set_title("Q-Q Plot of MEDV", fontweight="bold")
plt.tight_layout()
plt.savefig("01_target_distribution.png", dpi=120, bbox_inches="tight")
print("\n[OK]  Saved: 01_target_distribution.png")

# 3b. Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("02_correlation_heatmap.png", dpi=120, bbox_inches="tight")
print("[OK]  Saved: 02_correlation_heatmap.png")

# 3c. Top correlated features vs MEDV
top_feats = corr["MEDV"].drop("MEDV").abs().sort_values(ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, feat in enumerate(top_feats):
    axes[i].scatter(df[feat], df["MEDV"], alpha=0.4, color="#5C7CFA", edgecolors="none", s=20)
    m, b = np.polyfit(df[feat].fillna(df[feat].median()), df["MEDV"].fillna(df["MEDV"].median()), 1)
    x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
    axes[i].plot(x_line, m * x_line + b, color="#FF6B6B", linewidth=2)
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel("MEDV")
    corr_val = df[feat].corr(df["MEDV"])
    axes[i].set_title(f"{feat} vs MEDV  (r={corr_val:.2f})", fontweight="bold")
plt.suptitle("Top 6 Features vs House Price", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("03_feature_scatter_plots.png", dpi=120, bbox_inches="tight")
print("[OK]  Saved: 03_feature_scatter_plots.png")

# 3d. Missing values bar chart
fig, ax = plt.subplots(figsize=(8, 4))
missing_cols = missing[missing > 0]
bars = ax.bar(missing_cols.index, missing_cols.values, color="#F9844A", edgecolor="white")
for bar, val in zip(bars, missing_cols.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            str(val), ha="center", fontsize=10, fontweight="bold")
ax.set_title("Missing Value Counts per Feature", fontweight="bold")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("04_missing_values.png", dpi=120, bbox_inches="tight")
print("[OK]  Saved: 04_missing_values.png")

# ── 4. Preprocessing ──────────────────────────────────────────────────────────
print("\n\n--- Preprocessing ---")

# 4a. Separate features and target; drop rows where target is missing
df = df.dropna(subset=["MEDV"])
X = df.drop("MEDV", axis=1)
y = df["MEDV"]
print(f"  Rows after dropping missing MEDV: {len(df)}")

# 4b. Impute remaining missing values with median
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 4c. Train / Test split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)
print(f"  Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# 4d. Feature scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. Model Comparison (Cross-Validation) ────────────────────────────────────
print("\n\n--- 5-Fold Cross-Validation: Model Comparison ---")

models = {
    "Linear Regression"       : LinearRegression(),
    "Ridge"                   : Ridge(alpha=10),
    "Lasso"                   : Lasso(alpha=0.1),
    "ElasticNet"              : ElasticNet(alpha=0.1, l1_ratio=0.5),
    "KNN"                     : KNeighborsRegressor(n_neighbors=5),
    "Decision Tree"           : DecisionTreeRegressor(random_state=42),
    "Random Forest"           : RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting"       : GradientBoostingRegressor(n_estimators=200, random_state=42),
    "Extra Trees"             : ExtraTreesRegressor(n_estimators=200, random_state=42),
    "SVR"                     : SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train_sc, y_train,
                             cv=kf, scoring="r2")
    cv_results[name] = scores
    print(f"  {name:<26}  R² = {scores.mean():.4f} ± {scores.std():.4f}")

# ── 6. Model Comparison Plot ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
names    = list(cv_results.keys())
means    = [v.mean() for v in cv_results.values()]
stds     = [v.std()  for v in cv_results.values()]
colors   = ["#5C7CFA" if m < max(means) else "#F9844A" for m in means]
bars = ax.barh(names, means, xerr=stds, color=colors, edgecolor="white",
               height=0.6, capsize=4)
ax.axvline(0, color="white", linewidth=0.8)
for bar, m in zip(bars, means):
    ax.text(m + 0.005, bar.get_y() + bar.get_height()/2,
            f"{m:.3f}", va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Cross-Validated R² Score")
ax.set_title("Model Comparison — 5-Fold CV R² Score", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("05_model_comparison.png", dpi=120, bbox_inches="tight")
print("\n[OK]  Saved: 05_model_comparison.png")

# ── 7. Best Model — Train & Evaluate ─────────────────────────────────────────
best_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\n\n*  Best model: {best_name}")

best_model = models[best_name]
best_model.fit(X_train_sc, y_train)
y_pred = best_model.predict(X_test_sc)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\n--- Test Set Metrics ({best_name}) ---")
print(f"  MAE  : {mae:.4f}")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")

# ── 8. Evaluation Plots ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig)

# 8a. Actual vs Predicted
ax1 = fig.add_subplot(gs[0])
ax1.scatter(y_test, y_pred, alpha=0.6, color="#5C7CFA", edgecolors="none", s=30)
lim = [min(y_test.min(), y_pred.min()) - 1, max(y_test.max(), y_pred.max()) + 1]
ax1.plot(lim, lim, "--", color="#FF6B6B", linewidth=2, label="Perfect fit")
ax1.set_xlabel("Actual MEDV")
ax1.set_ylabel("Predicted MEDV")
ax1.set_title(f"Actual vs Predicted\nR² = {r2:.4f}", fontweight="bold")
ax1.legend()

# 8b. Residual plot
residuals = y_test - y_pred
ax2 = fig.add_subplot(gs[1])
ax2.scatter(y_pred, residuals, alpha=0.6, color="#F9844A", edgecolors="none", s=30)
ax2.axhline(0, color="#FF6B6B", linewidth=2, linestyle="--")
ax2.set_xlabel("Predicted MEDV")
ax2.set_ylabel("Residuals")
ax2.set_title("Residual Plot", fontweight="bold")

# 8c. Residual distribution
ax3 = fig.add_subplot(gs[2])
sns.histplot(residuals, kde=True, ax=ax3, color="#4ECDC4")
ax3.axvline(0, color="#FF6B6B", linewidth=2, linestyle="--")
ax3.set_title("Residual Distribution", fontweight="bold")
ax3.set_xlabel("Residuals")

plt.suptitle(f"Model Evaluation — {best_name}", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("06_evaluation_plots.png", dpi=120, bbox_inches="tight")
print("[OK]  Saved: 06_evaluation_plots.png")

# ── 9. Feature Importance (if available) ─────────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors_fi = ["#5C7CFA" if v < importances.max() else "#F9844A" for v in importances]
    importances.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="white")
    ax.set_title(f"Feature Importances — {best_name}", fontweight="bold", fontsize=13)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("07_feature_importances.png", dpi=120, bbox_inches="tight")
    print("[OK]  Saved: 07_feature_importances.png")

elif hasattr(best_model, "coef_"):
    coef = pd.Series(np.abs(best_model.coef_), index=X.columns)
    coef = coef.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    coef.plot(kind="barh", ax=ax, color="#5C7CFA", edgecolor="white")
    ax.set_title(f"Feature Coefficients (abs) — {best_name}", fontweight="bold", fontsize=13)
    ax.set_xlabel("|Coefficient|")
    plt.tight_layout()
    plt.savefig("07_feature_importances.png", dpi=120, bbox_inches="tight")
    print("[OK]  Saved: 07_feature_importances.png")

# ── 10. All Models Final Test Performance ─────────────────────────────────────
print("\n\n--- Final Test-Set Performance of ALL Models ---")
results = []
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    yp = model.predict(X_test_sc)
    results.append({
        "Model"  : name,
        "MAE"    : round(mean_absolute_error(y_test, yp), 4),
        "RMSE"   : round(np.sqrt(mean_squared_error(y_test, yp)), 4),
        "R²"     : round(r2_score(y_test, yp), 4)
    })

results_df = pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))
results_df.to_csv("model_results.csv", index=False)
print("\n[OK]  Saved: model_results.csv")

print("\n" + "=" * 65)
print("  Pipeline Complete!")
print(f"  Best Model : {best_name}")
print(f"  Test R²    : {r2:.4f}")
print(f"  Test RMSE  : {rmse:.4f}  ($1000s)")
print("=" * 65)

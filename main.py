"""
AQI Bucket Classification Pipeline
====================================
Dataset : city_day.csv  (Kaggle – Air Quality Data in India 2015-2020)
Run     : python main.py
Charts  : saved as PNG files in the same folder
"""

# ─── 0. IMPORTS ──────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # no display needed; saves PNGs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing       import LabelEncoder, StandardScaler
from sklearn.decomposition       import PCA
from sklearn.feature_selection   import SelectKBest, f_classif
from sklearn.linear_model        import LogisticRegression
from sklearn.ensemble            import (RandomForestClassifier,
                                          GradientBoostingClassifier,
                                          VotingClassifier)
from sklearn.metrics             import (accuracy_score, classification_report,
                                          confusion_matrix, roc_auc_score,
                                          ConfusionMatrixDisplay)
from sklearn.pipeline            import Pipeline

warnings.filterwarnings("ignore")

# ─── COLOUR PALETTE (colour-blind safe) ─────────────────────────────────────
BLUE   = "#2a78d6"
ORANGE = "#eb6834"
GREEN  = "#1baf7a"
GREY   = "#8a8985"
PAL    = [BLUE, ORANGE, GREEN, "#9b59b6", "#e74c3c", "#f39c12"]

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 – DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  SECTION 1 : DATA LOADING")
print("="*65)

CSV = "city_day.csv"

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
              "CO",    "SO2",  "O3", "Benzene", "Toluene", "Xylene"]
TARGET     = "AQI_Bucket"

if os.path.exists(CSV):
    df_raw = pd.read_csv(CSV, parse_dates=["Date"])
    print(f"Loaded  : {CSV}  →  {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
else:
    print("city_day.csv NOT FOUND – generating synthetic demo data …")
    rng   = np.random.default_rng(42)
    n     = 3000
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    cities = np.random.choice(["Delhi","Mumbai","Chennai","Kolkata","Bangalore"], n)

    pm25  = rng.gamma(4, 20, n)
    pm10  = pm25 * rng.uniform(1.4, 2.0, n)
    no2   = rng.gamma(2, 18, n)
    co    = rng.gamma(2,  1, n)
    so2   = rng.gamma(1.5, 8, n)
    o3    = rng.gamma(3, 12, n)
    nh3   = rng.gamma(2,  6, n)
    benz  = rng.gamma(1.5, 2, n)
    tol   = rng.gamma(2,   3, n)
    xyl   = rng.gamma(1.5, 2, n)
    no    = rng.gamma(2,  10, n)
    nox   = no + no2

    # Inject ~18 % missing values in several columns (messy data)
    for arr in [nh3, benz, tol, xyl, no, nox]:
        mask = rng.random(n) < 0.18
        arr[mask] = np.nan

    aqi = (0.4*pm25 + 0.2*pm10 + 0.15*no2 + 0.1*co*20
           + 0.08*so2 + 0.07*o3 + rng.normal(0, 15, n)).clip(0)

    def to_bucket(a):
        if   a <= 50:  return "Good"
        elif a <= 100: return "Satisfactory"
        elif a <= 200: return "Moderate"
        elif a <= 300: return "Poor"
        elif a <= 400: return "Very Poor"
        else:          return "Severe"

    buckets = [to_bucket(v) for v in aqi]

    df_raw = pd.DataFrame({
        "City":df_raw if False else pd.Series(cities),
        "Date":dates,
        "PM2.5":pm25, "PM10":pm10, "NO":no, "NO2":no2, "NOx":nox,
        "NH3":nh3, "CO":co, "SO2":so2, "O3":o3,
        "Benzene":benz, "Toluene":tol, "Xylene":xyl,
        "AQI":aqi, "AQI_Bucket":buckets
    })
    print(f"Demo data: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

print("\nColumn dtypes:\n", df_raw.dtypes)
print("\nFirst 3 rows:\n", df_raw.head(3).to_string())

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 – DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  SECTION 2 : DATA PREPROCESSING")
print("="*65)

df = df_raw.copy()

# ── 2a. Drop rows with no target ────────────────────────────────────────────
before = len(df)
df = df.dropna(subset=[TARGET, "AQI"])
print(f"\n[2a] Rows dropped (missing target): {before - len(df)}")

# ── 2b. Keep only pollutant columns that exist ──────────────────────────────
avail_poll = [c for c in POLLUTANTS if c in df.columns]
print(f"[2b] Pollutant features present  : {avail_poll}")

# ── 2c. Missing-value analysis ──────────────────────────────────────────────
miss = df[avail_poll].isnull().sum()
miss_pct = (miss / len(df) * 100).round(1)
print("\n[2c] Missing values per feature:")
for col, pct in miss_pct[miss_pct > 0].items():
    print(f"     {col:12s}  {pct:5.1f} %")

# ── 2d. Impute with MEDIAN (robust to outliers) ─────────────────────────────
medians = df[avail_poll].median()
df[avail_poll] = df[avail_poll].fillna(medians)
print(f"\n[2d] Imputed with column medians → 0 missing values remain")
assert df[avail_poll].isnull().sum().sum() == 0

# ── 2e. IQR outlier capping (Winsorization) ─────────────────────────────────
for col in avail_poll:
    Q1, Q3 = df[col].quantile([0.01, 0.99])
    iqr     = Q3 - Q1
    lo, hi  = Q1 - 1.5*iqr, Q3 + 1.5*iqr
    clipped = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lo, hi)
print(f"[2e] Outlier capping (1-99th pct) done")

# ── 2f. Encode target ───────────────────────────────────────────────────────
BUCKET_ORDER = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
present_buckets = [b for b in BUCKET_ORDER if b in df[TARGET].unique()]
le = LabelEncoder()
le.fit(present_buckets)
df["label"] = le.transform(df[TARGET])
print(f"\n[2f] AQI_Bucket classes → numeric label:")
for i, cls in enumerate(le.classes_):
    print(f"     {i}  →  {cls}  ({(df['label']==i).sum()} rows)")

# ── 2g. Date features ───────────────────────────────────────────────────────
df["month"]     = df["Date"].dt.month
df["day_of_yr"] = df["Date"].dt.dayofyear
df["quarter"]   = df["Date"].dt.quarter
print(f"\n[2g] Added date features: month, day_of_year, quarter")

print(f"\nAfter preprocessing: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 – FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  SECTION 3 : FEATURE ENGINEERING")
print("="*65)

# ── 3a. Pollutant ratio features ────────────────────────────────────────────
eps = 1e-6
if "NO2" in df.columns and "SO2" in df.columns:
    df["NO2_SO2_ratio"] = df["NO2"] / (df["SO2"] + eps)
if "PM2.5" in df.columns and "PM10" in df.columns:
    df["PM_ratio"]      = df["PM2.5"] / (df["PM10"] + eps)
if "CO" in df.columns and "O3" in df.columns:
    df["CO_O3_ratio"]   = df["CO"]   / (df["O3"]   + eps)
print("[3a] Ratio features added: NO2_SO2_ratio, PM_ratio, CO_O3_ratio")

# ── 3b. Lag features for day-ahead prediction ───────────────────────────────
df = df.sort_values(["City","Date"])
for lag in [1, 2, 3]:
    df[f"AQI_lag{lag}"] = df.groupby("City")["AQI"].shift(lag)
df = df.dropna(subset=[f"AQI_lag{l}" for l in [1,2,3]])
print(f"[3b] Lag features added (AQI_lag1, lag2, lag3); rows now: {len(df):,}")

# ── 3c. All features ────────────────────────────────────────────────────────
ratio_cols = [c for c in ["NO2_SO2_ratio","PM_ratio","CO_O3_ratio"] if c in df.columns]
date_cols  = ["month","day_of_yr","quarter"]
lag_cols   = ["AQI_lag1","AQI_lag2","AQI_lag3"]
feat_cols  = avail_poll + ratio_cols + date_cols + lag_cols
print(f"[3c] Total features: {len(feat_cols)}  →  {feat_cols}")

X = df[feat_cols].values
y = df["label"].values

# ── 3d. Train / test split (stratified 80/20) ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n[3d] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 3e. Standardise ─────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("[3e] Features standardised (zero mean, unit variance)")

# ── 3f. SelectKBest (ANOVA F-score) ─────────────────────────────────────────
K = min(10, len(feat_cols))
selector = SelectKBest(f_classif, k=K)
selector.fit(X_train, y_train)
scores   = pd.Series(selector.scores_, index=feat_cols).sort_values(ascending=False)
print(f"\n[3f] Top {K} features by ANOVA F-score:")
print(scores.head(K).to_string())

X_train_sel = selector.transform(X_train)
X_test_sel  = selector.transform(X_test)

# ── 3g. PCA visualisation (2 components on selected features) ───────────────
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train_sel)
print(f"\n[3g] PCA variance explained: PC1={pca.explained_variance_ratio_[0]*100:.1f}%"
      f"  PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

fig, ax = plt.subplots(figsize=(8, 5))
for i, cls in enumerate(le.classes_):
    mask = y_train == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               color=PAL[i % len(PAL)], alpha=0.45, s=18, label=cls)
ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("PCA – AQI Bucket Clusters (training set)")
ax.legend(title="AQI Bucket", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fig1_pca.png", dpi=130)
plt.close()
print("  → Saved fig1_pca.png")

# Feature importance bar (F-scores)
fig, ax = plt.subplots(figsize=(8, 4))
top_s = scores.head(K)
colors_bar = [BLUE if i % 2 == 0 else ORANGE for i in range(len(top_s))]
ax.barh(top_s.index[::-1], top_s.values[::-1], color=colors_bar[::-1])
ax.set_xlabel("ANOVA F-score")
ax.set_title("Feature Selection – Top Features")
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fig2_feature_importance.png", dpi=130)
plt.close()
print("  → Saved fig2_feature_importance.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 – MODEL BUILDING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  SECTION 4 : MODEL BUILDING")
print("="*65)

n_classes = len(le.classes_)

# ── 4a. Logistic Regression (baseline) ──────────────────────────────────────
print("\n[4a] Logistic Regression …")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sel, y_train)
print("     Done.")

# ── 4b. Random Forest ────────────────────────────────────────────────────────
print("[4b] Random Forest …")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sel, y_train)
print("     Done.")

# ── 4c. Gradient Boosting ────────────────────────────────────────────────────
print("[4c] Gradient Boosting …")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_sel, y_train)
print("     Done.")

# ── 4d. Voting Ensemble (soft) ───────────────────────────────────────────────
print("[4d] Soft Voting Ensemble (LR + RF + GB) …")
vot = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
    voting="soft")
vot.fit(X_train_sel, y_train)
print("     Done.")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 – HYPERPARAMETER TUNING (GridSearchCV on RF)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  SECTION 5 : HYPERPARAMETER TUNING  (GridSearchCV – Random Forest)")
print("="*65)

param_grid = {
    "n_estimators" : [100, 200],
    "max_depth"    : [None, 10, 20],
    "min_samples_split": [2, 5],
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0
)
grid.fit(X_train_sel, y_train)
print(f"\nBest params  : {grid.best_params_}")
print(f"Best CV acc  : {grid.best_score_*100:.2f} %")
rf_best = grid.best_estimator_

# ── Re-fit Voting with tuned RF ──────────────────────────────────────────────
vot_tuned = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf_best), ("gb", gb)],
    voting="soft")
vot_tuned.fit(X_train_sel, y_train)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 – MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  SECTION 6 : MODEL EVALUATION")
print("="*65)

def evaluate(name, model, Xtr, Xte, ytr, yte):
    y_pred  = model.predict(Xte)
    y_prob  = model.predict_proba(Xte)
    acc     = accuracy_score(yte, y_pred)
    report  = classification_report(yte, y_pred,
                                     target_names=le.classes_,
                                     output_dict=True,
                                     zero_division=0)
    macro   = report["macro avg"]
    try:
        if y_prob.shape[1] > 2:
            auc = roc_auc_score(yte, y_prob,
                                multi_class="ovr", average="macro")
        else:
            auc = roc_auc_score(yte, y_prob[:, 1])
    except Exception:
        auc = float("nan")

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Accuracy          : {acc*100:6.2f} %")
    print(f"  Macro Precision   : {macro['precision']*100:6.2f} %")
    print(f"  Macro Recall      : {macro['recall']*100:6.2f} %")
    print(f"  Macro F1          : {macro['f1-score']*100:6.2f} %")
    print(f"  ROC-AUC (OvR)     : {auc:.4f}")
    return {"Model": name, "Accuracy": acc,
            "Precision": macro["precision"],
            "Recall":    macro["recall"],
            "F1":        macro["f1-score"],
            "AUC":       auc}, y_pred

results = []
preds   = {}

for name, mdl in [("Logistic Regression", lr),
                   ("Random Forest",       rf),
                   ("Gradient Boosting",   gb),
                   ("Voting Ensemble",     vot_tuned)]:
    r, p = evaluate(name, mdl, X_train_sel, X_test_sel, y_train, y_test)
    results.append(r)
    preds[name] = p

res_df = pd.DataFrame(results).set_index("Model")
print("\n\n  ── SUMMARY TABLE ──")
print(res_df.map(lambda x: f"{x*100:.2f}%").to_string())

# ── Confusion matrix for best model (Voting Ensemble) ───────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, preds["Voting Ensemble"])
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix – Voting Ensemble")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("fig3_confusion_matrix.png", dpi=130)
plt.close()
print("\n  → Saved fig3_confusion_matrix.png")

# ── Model comparison bar chart ───────────────────────────────────────────────
metrics_plot = ["Accuracy","Precision","Recall","F1","AUC"]
fig, axes = plt.subplots(1, len(metrics_plot), figsize=(14, 4), sharey=False)
for i, met in enumerate(metrics_plot):
    vals = res_df[met].values * 100
    colors_m = [BLUE, ORANGE, GREEN, "#9b59b6"]
    axes[i].bar(range(len(res_df)), vals, color=colors_m)
    axes[i].set_xticks(range(len(res_df)))
    axes[i].set_xticklabels([m.replace(" ","\n") for m in res_df.index], fontsize=7)
    axes[i].set_title(met, fontsize=9)
    axes[i].set_ylim(0, 105)
    axes[i].grid(axis="y", linestyle="--", alpha=0.4)
    for j, v in enumerate(vals):
        axes[i].text(j, v + 1, f"{v:.1f}", ha="center", fontsize=7)
fig.suptitle("Model Comparison – All Metrics (%)", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("fig4_model_comparison.png", dpi=130)
plt.close()
print("  → Saved fig4_model_comparison.png")

# ── Feature importance from tuned RF ────────────────────────────────────────
selected_names = [feat_cols[i] for i in selector.get_support(indices=True)]
fi = pd.Series(rf_best.feature_importances_, index=selected_names).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
fi.plot(kind="barh", color=BLUE, ax=ax)
ax.set_title("Random Forest – Feature Importances (tuned)")
ax.set_xlabel("Mean Decrease in Impurity")
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fig5_rf_feature_importance.png", dpi=130)
plt.close()
print("  → Saved fig5_rf_feature_importance.png")

# AQI bucket distribution (class balance)
fig, ax = plt.subplots(figsize=(7, 4))
counts = pd.Series(le.inverse_transform(y), name="AQI_Bucket").value_counts()
counts = counts.reindex([b for b in BUCKET_ORDER if b in counts.index])
ax.bar(counts.index, counts.values,
       color=[PAL[i % len(PAL)] for i in range(len(counts))])
ax.set_title("Class Distribution – AQI Buckets")
ax.set_ylabel("Count")
ax.set_xlabel("AQI Bucket")
plt.xticks(rotation=20, ha="right")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fig6_class_distribution.png", dpi=130)
plt.close()
print("  → Saved fig6_class_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 – TOMORROW'S AQI BUCKET PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  SECTION 7 : TOMORROW'S AQI BUCKET PREDICTION")
print("="*65)
print("""
How it works:
  The model was trained with lag features (AQI_lag1, lag2, lag3).
  Given TODAY's pollutant readings + the last 3 days' AQI values,
  it predicts TOMORROW's AQI bucket.
""")

# Use last known row from each city as "today"
last_rows = (df.sort_values("Date")
               .groupby("City")
               .tail(1)
               .reset_index(drop=True))

print(f"{'City':<14} {'Today AQI':>10} {'Predicted Tomorrow':>20}")
print("─" * 46)
for _, row in last_rows.iterrows():
    sample = row[feat_cols].values.reshape(1, -1)
    sample_sc  = scaler.transform(sample)
    sample_sel = selector.transform(sample_sc)
    pred_label = vot_tuned.predict(sample_sel)[0]
    pred_prob  = vot_tuned.predict_proba(sample_sel)[0]
    pred_class = le.inverse_transform([pred_label])[0]
    conf       = pred_prob[pred_label] * 100
    today_aqi  = row["AQI"]
    print(f"{row['City']:<14} {today_aqi:>10.1f}   {pred_class:<15}  ({conf:.0f}% conf)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  charts saved as PNG files in the current folder")
print("="*65)
print("""
Files created:
  fig1_pca.png                 - PCA cluster plot (class separation)
  fig2_feature_importance.png  - ANOVA F-score feature ranking
  fig3_confusion_matrix.png    - Confusion matrix (Voting Ensemble)
  fig4_model_comparison.png    - Accuracy / Precision / Recall / F1 / AUC
  fig5_rf_feature_importance.png - Random Forest feature importances
  fig6_class_distribution.png  - Class balance
""")

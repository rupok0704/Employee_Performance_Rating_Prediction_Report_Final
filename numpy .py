# ============================================================
# Employee Performance Rating Predictor  (Random Forest + RMSE)
# Copy this whole cell into Google Colab and press Run.
# ============================================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

FEATURES = ["Experience", "KPI_Score", "Training_Hrs", "Attendance", "Projects", "Peer_Feedback"]
TARGET = "Performance_Rating"
USE_OWN_CSV = False   # set True to upload the real Kaggle HR CSV (needs the columns above)

# ---------- 1. COLLECT ----------
def make_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    exp = np.clip(rng.gamma(3.0, 3.0, n), 0, 20).round(1)
    kpi = np.clip(rng.normal(70, 13, n) + exp * 0.4, 30, 100).round(0)
    train = np.clip(rng.normal(40, 18, n), 0, 80).round(0)
    att = np.clip(rng.normal(92, 6, n), 60, 100).round(0)
    proj = np.clip(rng.poisson(7, n) + 1, 1, 14)
    peer = np.clip(rng.normal(3.5, 0.8, n), 1, 5).round(1)
    r = (1 + 0.030*(kpi-30) + 0.060*exp + 0.20*(peer-1) + 0.008*train
         + 0.004*(att-60) + 0.010*proj + rng.normal(0, 0.25, n))
    df = pd.DataFrame({"Experience": exp, "KPI_Score": kpi, "Training_Hrs": train,
                       "Attendance": att, "Projects": proj, "Peer_Feedback": peer,
                       TARGET: np.clip(r, 1, 5).round(2)})
    for c in ["KPI_Score", "Training_Hrs", "Peer_Feedback"]:      # a few missing values
        df.loc[rng.choice(n, 12, replace=False), c] = np.nan
    return df

if USE_OWN_CSV:
    from google.colab import files
    up = files.upload()
    df = pd.read_csv(next(iter(up)))
else:
    df = make_data()
print("Dataset shape:", df.shape)

# ---------- 2. CLEAN ----------
df = df.drop_duplicates()
df[df.select_dtypes("number").columns] = df.select_dtypes("number").fillna(df.select_dtypes("number").median())
for c in FEATURES:                                   # IQR outlier clipping
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    df[c] = df[c].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
df[TARGET] = df[TARGET].clip(1, 5)

# ---------- 3. EXPLORE ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].hist(df[TARGET], bins=20, color="#d98206", edgecolor="white")
ax[0].set(title="Rating distribution", xlabel="Rating", ylabel="Employees")
corr = df[FEATURES + [TARGET]].corr()
im = ax[1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax[1].set_xticks(range(len(corr))); ax[1].set_xticklabels(corr.columns, rotation=45, ha="right")
ax[1].set_yticks(range(len(corr))); ax[1].set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax[1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax[1].set_title("Correlation heatmap"); fig.colorbar(im, ax=ax[1]); plt.tight_layout(); plt.show()

# ---------- 4. PREPARE (80/20 split) ----------
X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df[TARGET], test_size=0.2, random_state=42)
print(f"Train: {len(X_tr)}  Test: {len(X_te)}")

# ---------- 5. TRAIN ----------
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)

# ---------- 6. EVALUATE ----------
pred = model.predict(X_te)
rmse = np.sqrt(mean_squared_error(y_te, pred))
print(f"\nTest RMSE: {rmse:.3f} | R2: {r2_score(y_te, pred):.3f} | MAE: {mean_absolute_error(y_te, pred):.3f}")

sizes = [10, 25, 50, 100, 150, 200]
sweep = [np.sqrt(mean_squared_error(y_te, RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
         .fit(X_tr, y_tr).predict(X_te))) for n in sizes]
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].plot(sizes, sweep, marker="o", color="#4c2f9e"); ax[0].set(title="RMSE vs number of trees", xlabel="Trees", ylabel="Test RMSE"); ax[0].grid(alpha=.3)
ax[1].barh(imp.index, imp.values, color="#0f9d8e"); ax[1].set_title("Feature importance")
ax[2].scatter(y_te, pred, s=12, alpha=.6, color="#4c2f9e"); ax[2].plot([1, 5], [1, 5], "r--")
ax[2].set(title="Actual vs predicted", xlabel="Actual", ylabel="Predicted")
plt.tight_layout(); plt.show()

# ---------- 7. PREDICT ----------
MEDIANS = dict(Experience=10, KPI_Score=70, Training_Hrs=40, Attendance=92, Projects=8, Peer_Feedback=3)

def level(r):
    return ("Needs improvement" if r < 2 else "Developing" if r < 3 else
            "Meets expectations" if r < 3.75 else "Exceeds expectations" if r < 4.3 else "Outstanding")

def predict_rating(**kw):
    x = pd.DataFrame([[kw[f] for f in FEATURES]], columns=FEATURES)
    return float(np.clip(model.predict(x)[0], 1, 5))

def predict_employee(Experience=6, KPI_Score=78, Training_Hrs=40, Attendance=94, Projects=8, Peer_Feedback=4.2):
    row = dict(Experience=Experience, KPI_Score=KPI_Score, Training_Hrs=Training_Hrs,
               Attendance=Attendance, Projects=Projects, Peer_Feedback=Peer_Feedback)
    r = predict_rating(**row)
    print(f"Predicted rating: {r:.2f} / 5  ->  {level(r)}\nWhy this rating? (effect vs a typical employee)")
    for f in FEATURES:
        alt = dict(row); alt[f] = MEDIANS[f]
        print(f"  {f:<14}{r - predict_rating(**alt):+.2f}")

predict_employee()                      # example; edit the numbers and re-run

# ---------- 8. INTERACTIVE SLIDERS (like the HTML app) ----------
try:
    import ipywidgets as w
    from IPython.display import display
    sliders = dict(
        Experience=w.FloatSlider(value=6, min=0, max=20, step=.5, description="Experience", style={"description_width": "110px"}),
        KPI_Score=w.IntSlider(value=70, min=30, max=100, description="KPI score", style={"description_width": "110px"}),
        Training_Hrs=w.IntSlider(value=40, min=0, max=80, description="Training hrs", style={"description_width": "110px"}),
        Attendance=w.IntSlider(value=92, min=60, max=100, description="Attendance %", style={"description_width": "110px"}),
        Projects=w.IntSlider(value=7, min=1, max=14, description="Projects", style={"description_width": "110px"}),
        Peer_Feedback=w.FloatSlider(value=3.5, min=1, max=5, step=.1, description="Peer feedback", style={"description_width": "110px"}))
    w.interact(predict_employee, **sliders)
except ImportError:
    print("ipywidgets not found - use predict_employee(...) above instead.")
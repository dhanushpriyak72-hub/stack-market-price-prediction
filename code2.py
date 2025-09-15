"""
Health Monitoring using Wearable Data (end-to-end)
Author: (you)

How to run:
1) pip install -r requirements.txt
2) python health_monitoring.py
   - The script will generate sample data if no CSV path is provided.
   - It will produce plots in ./outputs and a short report CSV.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------------------------
# Config & helpers
# ---------------------------
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
plt.style.use("ggplot")   # or "seaborn-v0_8-whitegrid"



def generate_sample_data(path="sample_wearable.csv", days=90, seed=42):
    """Generate a sample wearable dataset (minute-level aggregated to daily)."""
    np.random.seed(seed)
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    dates = pd.date_range(start=start, periods=days, freq="D")

    rows = []
    for d in dates:
        # synthetic daily signals with realistic ranges + some randomness
        steps = int(np.clip(np.random.normal(8500, 2600), 800, 20000))
        avg_hr = int(np.clip(np.random.normal(72 - 0.01 * (steps - 8000) / 100, 6), 48, 110))
        sleep_hours = float(np.clip(np.random.normal(7.0 - 0.001*(steps-9000)/100, 1.1), 3.0, 9.5))
        calories = int(np.clip(2000 + 0.04 * steps + np.random.normal(0, 120), 1200, 4000))
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "steps": steps,
            "avg_hr": avg_hr,
            "sleep_hours": round(sleep_hours, 2),
            "calories": calories
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Sample data saved to {path}")
    return path


def load_data(csv_path=None):
    """Load wearable CSV or generate sample if csv_path is None."""
    if csv_path is None:
        csv_path = generate_sample_data()
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # ensure types
    for col in ["steps", "avg_hr", "sleep_hours", "calories"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------
# Preprocessing & features
# ---------------------------
def preprocess(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").asfreq("D")
    # Fill short gaps with interpolation; leave longer as NaN
    df["steps"] = df["steps"].interpolate(limit=2).fillna(method="ffill").fillna(method="bfill")
    df["avg_hr"] = df["avg_hr"].interpolate(limit=2).fillna(method="ffill").fillna(method="bfill")
    df["sleep_hours"] = df["sleep_hours"].interpolate(limit=2).fillna(method="ffill").fillna(method="bfill")
    df["calories"] = df["calories"].interpolate(limit=2).fillna(method="ffill").fillna(method="bfill")

    # Rolling features
    df["steps_7d_mean"] = df["steps"].rolling(7, min_periods=1).mean()
    df["hr_7d_mean"] = df["avg_hr"].rolling(7, min_periods=1).mean()
    df["sleep_7d_mean"] = df["sleep_hours"].rolling(7, min_periods=1).mean()

    # Personal baseline: rolling median and MAD
    df["steps_median_30d"] = df["steps"].rolling(30, min_periods=7).median()
    df["hr_median_30d"] = df["avg_hr"].rolling(30, min_periods=7).median()

    return df


# ---------------------------
# Exploratory Data Analysis
# ---------------------------
def plot_time_series(df):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    df["steps"].plot(ax=axes[0], title="Daily Steps")
    axes[0].set_ylabel("Steps")
    df["avg_hr"].plot(ax=axes[1], title="Avg Heart Rate (BPM)")
    axes[1].set_ylabel("BPM")
    df["sleep_hours"].plot(ax=axes[2], title="Sleep Hours")
    axes[2].set_ylabel("Hours")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "timeseries_steps_hr_sleep.png")
    fig.savefig(out, dpi=150)
    print("Saved:", out)
    plt.close(fig)


def plot_correlation(df):
    corr = df[["steps", "avg_hr", "sleep_hours", "calories"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    out = os.path.join(OUT_DIR, "correlation_matrix.png")
    fig.savefig(out, dpi=150)
    print("Saved:", out)
    plt.close(fig)


def plot_seasonality(df, column="steps"):
    # show weekly seasonality by day-of-week boxplot
    tmp = df.reset_index()
    tmp["dow"] = tmp["date"].dt.day_name()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(data=tmp, x="dow", y=column, order=order, ax=ax)
    ax.set_title(f"{column} by Day of Week")
    out = os.path.join(OUT_DIR, f"{column}_by_dow.png")
    fig.savefig(out, dpi=150)
    print("Saved:", out)
    plt.close(fig)


# ---------------------------
# Anomaly detection
# ---------------------------
def detect_anomalies(df, field, window=14, z_thresh=2.5):
    """
    Rolling baseline z-score anomaly detection.
    Returns dataframe with anomaly boolean series named '{field}_anomaly'
    """
    s = df[field]
    rolling_mean = s.rolling(window, min_periods=3).mean()
    rolling_std = s.rolling(window, min_periods=3).std().replace(0, np.nan)
    z = (s - rolling_mean) / rolling_std
    anomaly_col = f"{field}_anomaly"
    df[anomaly_col] = (z.abs() > z_thresh)
    return df


def plot_anomalies(df, field):
    col_anom = f"{field}_anomaly"
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df.index, df[field], label=field)
    ax.scatter(df.index[df[col_anom]], df[field][df[col_anom]], color="red", label="anomaly", zorder=5)
    ax.set_title(f"Anomalies in {field}")
    ax.legend()
    out = os.path.join(OUT_DIR, f"anomalies_{field}.png")
    fig.savefig(out, dpi=150)
    print("Saved:", out)
    plt.close(fig)


# ---------------------------
# Predictive model: next-day steps (simple supervised)
# ---------------------------
def create_lag_features(df, target="steps", lags=(1,2,3,7,14)):
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    # add rolling stats
    df[f"{target}_roll_mean_7"] = df[target].rolling(7).mean().shift(1)
    df[f"{target}_roll_std_7"] = df[target].rolling(7).std().shift(1)
    df = df.dropna()
    return df


def train_predict_steps(df):
    df_feat = create_lag_features(df, target="steps")
    features = [c for c in df_feat.columns if c.startswith("steps_") and c != "steps"]
    # also include hr & sleep 1-day lag
    df_feat["avg_hr_lag1"] = df_feat["avg_hr"].shift(1)
    df_feat["sleep_hours_lag1"] = df_feat["sleep_hours"].shift(1)
    df_feat = df_feat.dropna()
    features += ["avg_hr_lag1", "sleep_hours_lag1"]

    X = df_feat[features]
    y = df_feat["steps"]

    # train/test split chronologically
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Step prediction performance:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2:", r2_score(y_test, y_pred))

    # Save predictions to CSV
    res = pd.DataFrame({
        "date": X_test.index,
        "steps_actual": y_test.values,
        "steps_pred": np.round(y_pred).astype(int)
    }).set_index("date")
    res.to_csv(os.path.join(OUT_DIR, "steps_predictions.csv"))
    print("Saved predictions:", os.path.join(OUT_DIR, "steps_predictions.csv"))

    # plot
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(res.index, res["steps_actual"], label="actual")
    ax.plot(res.index, res["steps_pred"], label="predicted", alpha=0.8)
    ax.set_title("Actual vs Predicted Steps (test set)")
    ax.legend()
    out = os.path.join(OUT_DIR, "steps_actual_vs_predicted.png")
    fig.savefig(out, dpi=150)
    print("Saved:", out)
    plt.close(fig)

    return model


# ---------------------------
# Simple decomposition (seasonal/trend)
# ---------------------------
def decompose_series(df, column="steps"):
    s = df[column].dropna()
    if len(s) < 14:
        print("Not enough points for decomposition")
        return
    res = seasonal_decompose(s, model="additive", period=7, two_sided=False)
    fig = res.plot()
    fig.set_size_inches(10,6)
    out = os.path.join(OUT_DIR, f"decompose_{column}.png")
    fig.savefig(out, dpi=150)
    print("Saved:", out)
    plt.close(fig)


# ---------------------------
# Main flow
# ---------------------------
def main(csv_path=None):
    print("Loading data...")
    df = load_data(csv_path)
    print(df.head())

    print("Preprocessing & feature extraction...")
    df = preprocess(df)

    print("Plotting time series...")
    plot_time_series(df)
    plot_correlation(df)
    plot_seasonality(df, "steps")

    print("Decomposing steps series...")
    decompose_series(df, "steps")

    # Anomaly detection on avg_hr and sleep_hours
    print("Detecting anomalies...")
    df = detect_anomalies(df, "avg_hr", window=14, z_thresh=2.8)
    df = detect_anomalies(df, "sleep_hours", window=14, z_thresh=2.5)
    plot_anomalies(df, "avg_hr")
    plot_anomalies(df, "sleep_hours")

    # Save anomaly report
    anomalies = df[(df["avg_hr_anomaly"] == True) | (df["sleep_hours_anomaly"] == True)][[
        "steps", "avg_hr", "sleep_hours", "avg_hr_anomaly", "sleep_hours_anomaly"
    ]]
    anomalies.to_csv(os.path.join(OUT_DIR, "anomaly_report.csv"))
    print("Saved anomaly report:", os.path.join(OUT_DIR, "anomaly_report.csv"))

    # Predict next-day steps
    print("Training predictive model for next-day steps...")
    model = train_predict_steps(df)

    print("Done. Outputs saved in:", OUT_DIR)
    print("You can replace sample CSV with your wearable export and re-run using:")
    print("  python health_monitoring.py path/to/your_wearable.csv")


if __name__ == "__main__":
    import sys
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(csv_arg)

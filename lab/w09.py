import sys
import numpy as np
import pandas as pd

# Load dataset 
df = pd.read_csv("ccpp.csv")

# Section 3: Data-quality checks
print("--- Section 3: Data-quality checks ---")
print(f"Rows: {len(df)}, Columns: {len(df.columns)} ({', '.join(df.columns)})")
print(f"Missing values: {int(df.isnull().sum().sum())}")
print(f"Exact duplicate rows: {int(df.duplicated().sum())}")
for col, unit in [("AT", "°C"), ("V", "cm Hg"), ("AP", "mbar"), ("RH", "%"), ("PE", "MW")]:
    print(f"{col}: {df[col].min():.2f} - {df[col].max():.2f} {unit}")

# Section 4: Summary statistics and correlation
print("\n--- Section 4: EDA ---")
for col, unit in [("AT", "°C"), ("PE", "MW")]:
    print(f"{col} mean/median/std: {df[col].mean():.2f} / {df[col].median():.2f} / {df[col].std():.2f} {unit}")

corr = df.corr()["PE"]
print("Correlation with PE (r): " + ", ".join(f"{c}={corr[c]:.3f}" for c in ["AT", "V", "AP", "RH"]))

# Section 5: Models and test error
print("\n--- Section 5: Models and test error ---")
train = df.iloc[:7654]
test = df.iloc[7654:]
print(f"Split: {len(train)} train, {len(test)} test")

train_mean = train["PE"].mean()
baseline_mae = np.mean(np.abs(test["PE"] - train_mean))
print(f"Baseline: train mean = {train_mean:.2f} MW, test MAE = {baseline_mae:.2f} MW")

a_at, b_at = np.polyfit(train["AT"], train["PE"], 1)
mae_at = np.mean(np.abs(test["PE"] - (a_at * test["AT"] + b_at)))
print(f"Line (AT): a = {a_at:.3f} MW/°C, b = {b_at:.2f} MW, test MAE = {mae_at:.2f} MW")

a_v, b_v = np.polyfit(train["V"], train["PE"], 1)
mae_v = np.mean(np.abs(test["PE"] - (a_v * test["V"] + b_v)))
print(f"Line (V): a = {a_v:.3f} MW/cm Hg, b = {b_v:.2f} MW, test MAE = {mae_v:.2f} MW")

# 5.5 Hand-check row (first test row = file line 7656)
v_val = test.iloc[0]["V"]
true_pe = test.iloc[0]["PE"]
pred_pe = a_v * v_val + b_v
abs_err = abs(true_pe - pred_pe)
print(f"Row 7656: V = {v_val:.2f} cm Hg, pred = {pred_pe:.2f} MW, true = {true_pe:.2f} MW, |err| = {abs_err:.2f} MW")

print(f"\nVersions: Python {sys.version.split()[0]}, NumPy {np.__version__}, pandas {pd.__version__}")

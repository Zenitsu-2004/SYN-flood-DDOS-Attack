import pandas as pd
import numpy as np

# Load clean data
data = pd.read_csv(r"D:\PROJECT\data\train.csv")

print("===== ORIGINAL MISSING VALUES =====")
print(data.isnull().sum().sum())

# ----------------------------
# SIMULATE missing values
# ----------------------------
print("\nSimulating missing values ...")

data.iloc[0, 0] = np.nan
data.iloc[1, 1] = np.nan

print("\n===== MISSING VALUES AFTER SIMULATION =====")
print(data.isnull().sum()[data.isnull().sum() > 0])

# ----------------------------
# HANDLE missing values
# ----------------------------
print("\nHandling missing values using mean imputation...")

data.fillna(data.mean(numeric_only=True), inplace=True)

print("\n===== MISSING VALUES AFTER HANDLING =====")
print(data.isnull().sum().sum())

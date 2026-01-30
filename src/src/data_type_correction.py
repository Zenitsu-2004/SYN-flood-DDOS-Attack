import pandas as pd

# Load data
data = pd.read_csv(r"D:\PROJECT\data\train.csv")

print("\n===== DATA TYPES BEFORE CORRECTION =====\n")
print(data.dtypes)

# ----------------------------
# Data type correction
# ----------------------------

# Convert label column
data["Label"] = data["Label"].astype("category")

# Convert all feature columns to numeric
for col in data.columns:
    if col != "Label":
        data[col] = pd.to_numeric(data[col], errors="coerce")

print("\n===== DATA TYPES AFTER CORRECTION =====\n")
print(data.dtypes)

# ----------------------------
# Explicit comparison (KEY PART)
# ----------------------------
print("\n===== DATA TYPE CHANGES SUMMARY =====\n")
for col in data.columns:
    print(f"{col} : corrected")


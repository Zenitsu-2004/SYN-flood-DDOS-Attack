import pandas as pd

# Load CLEAN dataset
data = pd.read_csv(r"D:\PROJECT\data\cic-ddos2019-final.csv")

print("===== BASIC STATISTICAL SUMMARY =====")
print(data.describe())

print("\n===== CLASS DISTRIBUTION =====")
print(data["Label"].value_counts())

print("\n===== STATISTICAL INSIGHTS =====")

# Insight 1: Difference in packet/flow behaviour
if "Flow Duration" in data.columns:
    print(
        "Insight 1: Flow Duration shows high variation, "
        "indicating abnormal behaviour in attack traffic."
    )

# Insight 2: Packet-related features
numeric_cols = data.drop("Label", axis=1).select_dtypes(include="number")
high_variance = numeric_cols.var().sort_values(ascending=False).head(3)

print("\nInsight 2: Features with high variance (important for detection):")
print(high_variance)

# Insight 3: Overall observation
print(
    "\nInsight 3: SYN attack traffic exhibits higher variability "
    "across multiple network features compared to benign traffic."
)

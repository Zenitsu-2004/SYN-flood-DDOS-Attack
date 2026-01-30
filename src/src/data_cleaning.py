import pandas as pd
import numpy as np

# Step 1: Load raw dataset
df = pd.read_csv("data/cic-ddos2019-raw.csv")

print("Original dataset shape:", df.shape)

# Step 2: Remove duplicate rows
duplicate_count = df.duplicated().sum()
print("Duplicate rows found:", duplicate_count)

df.drop_duplicates(inplace=True)

# Step 3: Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Step 4: Remove rows with NaN values
missing_count = df.isnull().sum().sum()
print("Missing values found:", missing_count)

df.dropna(inplace=True)

print("Cleaned dataset shape:", df.shape)

# Step 5: Save cleaned dataset
df.to_csv("data/cic-ddos2019-clean.csv", index=False)

print("✅ Data cleaning completed and saved successfully!")

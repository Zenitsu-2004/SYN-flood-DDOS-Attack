import pandas as pd

# Load training data (update path if needed)
data = pd.read_csv(r"D:\PROJECT\data\train.csv")

print("===== DATA SHAPE =====")
print("Rows, Columns:", data.shape)

print("\n===== COLUMN NAMES =====")
print(data.columns)

print("\n===== DATA TYPES =====")
print(data.dtypes)

print("\n===== MISSING VALUES =====")
print(data.isnull().sum())

print("\n===== BASIC STATISTICAL SUMMARY =====")
print(data.describe())

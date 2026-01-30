import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# -------------------------------
# Step 1: Load RAW dataset
# -------------------------------
df = pd.read_csv("data/cic-ddos2019-raw.csv")
print("Raw dataset loaded:", df.shape)

# -------------------------------
# Step 2: Remove duplicates & invalid values
# -------------------------------
df.drop_duplicates(inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("After cleaning:", df.shape)

# -------------------------------
# Step 3: Encode Label
# -------------------------------
df['Label'] = df['Label'].astype(str).str.lower().str.strip()
df['Label'] = df['Label'].map({
    'benign': 0,
    'syn': 1
})

print("Label encoding done")
print(df['Label'].value_counts())

# -------------------------------
# Step 4: Select CRUCIAL FEATURES
# -------------------------------
crucial_features = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Packet Length Mean',
    'Packet Length Std',
    'Average Packet Size',
    'SYN Flag Count',
    'ACK Flag Count',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'Label'
]

# Keep only existing columns (safe check)
df_final = df[[col for col in crucial_features if col in df.columns]]

print("Final selected features:", df_final.shape)

# -------------------------------
# Step 5: Save FINAL cleaned dataset
# -------------------------------
os.makedirs("data", exist_ok=True)
df_final.to_csv("data/cic-ddos2019-final.csv", index=False)

print("✅ Final cleaned dataset saved")

# -------------------------------
# Step 6: Train-Test Split
# -------------------------------
X = df_final.drop('Label', axis=1)
y = df_final['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Combine back for saving
train_df = X_train.copy()
train_df['Label'] = y_train

test_df = X_test.copy()
test_df['Label'] = y_test

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("✅ Train/Test split completed")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

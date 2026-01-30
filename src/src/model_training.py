import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Create models folder
os.makedirs("models", exist_ok=True)

# ----------------------------------
# Load TRAIN dataset (FINAL FEATURES)
# ----------------------------------
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")

X_train = train_df.drop("Label", axis=1)
y_train = train_df["Label"]

X_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ----------------------------------
# Train Random Forest model
# ----------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ Model trained using FINAL features")

# ----------------------------------
# Evaluate
# ----------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------
# Save model
# ----------------------------------
joblib.dump(model, "models/ddos_random_forest_model.pkl")
print("✅ Model saved successfully")

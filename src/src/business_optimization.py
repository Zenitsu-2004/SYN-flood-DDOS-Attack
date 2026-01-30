import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

# Load test data
test_df = pd.read_csv("data/test.csv")
X_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"]

# Load tuned model
model = joblib.load("models/ddos_rf_tuned_model.pkl")

# Get probabilities for SYN attack (class 1)
y_proba = model.predict_proba(X_test)[:, 1]

# Default threshold
default_threshold = 0.5
y_pred_default = (y_proba >= default_threshold).astype(int)

# Business-optimized threshold
optimized_threshold = 0.3
y_pred_optimized = (y_proba >= optimized_threshold).astype(int)

print("===== BUSINESS OPTIMIZATION =====")

print("\nConfusion Matrix (Threshold = 0.5):")
print(confusion_matrix(y_test, y_pred_default))

print("\nConfusion Matrix (Threshold = 0.3):")
print(confusion_matrix(y_test, y_pred_optimized))

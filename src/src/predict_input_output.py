import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/ddos_random_forest_model.pkl")

# Load test data
test_df = pd.read_csv("data/test.csv")

# Separate features and label
X_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"]

# Take ONE sample as input
sample_index = 5
sample_input = X_test.iloc[[sample_index]]
actual_label = y_test.iloc[sample_index]

# Predict
prediction = model.predict(sample_input)[0]
probability = model.predict_proba(sample_input)[0][1]

# Convert to readable form
predicted_label = "SYN Attack" if prediction == 1 else "Benign"

print("===== INPUT → OUTPUT DEMO =====")
print("Sample Index:", sample_index)
print("Actual Label   :", actual_label)
print("Predicted Label:", predicted_label)
print("Threat Score   :", round(probability, 2))

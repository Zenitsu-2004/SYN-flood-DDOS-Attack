import pandas as pd
import joblib

# Load data to get feature names
data = pd.read_csv("data/cic-ddos2019-final.csv")
X = data.drop("Label", axis=1)

# Load tuned model
model = joblib.load("models/ddos_rf_tuned_model.pkl")

# Get feature importance
importance = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("===== FEATURE IMPORTANCE (TOP 10) =====")
print(feature_importance_df.head(10))

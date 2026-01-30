import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
data = pd.read_csv("data/cic-ddos2019-final.csv")
X = data.drop("Label", axis=1)
y = data["Label"]

# Baseline model
baseline_model = LogisticRegression(max_iter=1000)

# Tuned model
tuned_model = joblib.load("models/ddos_rf_tuned_model.pkl")

# Cross-validation scores
baseline_scores = cross_val_score(
    baseline_model, X, y, cv=3, scoring="f1"
)

tuned_scores = cross_val_score(
    tuned_model, X, y, cv=3, scoring="f1"
)

print("===== STATISTICAL VALIDATION =====")
print("Baseline Model F1 scores:", baseline_scores)
print("Baseline Mean F1:", baseline_scores.mean())

print("\nTuned Model F1 scores:", tuned_scores)
print("Tuned Mean F1:", tuned_scores.mean())

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# Load clean dataset
data = pd.read_csv("data/cic-ddos2019-final.csv")

X = data.drop("Label", axis=1)
y = data["Label"]

# Base model
rf = RandomForestClassifier(random_state=42)

# Parameters to tune (keep small for demo)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5]
}

# Grid Search with 3-fold CV
grid = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

print("Starting hyperparameter tuning...")
grid.fit(X, y)

# Best model
best_model = grid.best_estimator_

print("\nBest Parameters Found:")
print(grid.best_params_)

# Save model
joblib.dump(best_model, "models/ddos_rf_tuned_model.pkl")

print("\nTuned model saved successfully.")

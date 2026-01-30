import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load cleaned data
train_df = pd.read_csv("data/train_corrected.csv")
test_df = pd.read_csv("data/test.csv")

X_train = train_df.drop("Label", axis=1)
y_train = train_df["Label"]

X_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"]

# Advanced Random Forest (with imbalance handling)
advanced_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

advanced_model.fit(X_train, y_train)

# Predictions
y_pred = advanced_model.predict(X_test)
y_prob = advanced_model.predict_proba(X_test)[:, 1]

print("===== ADVANCED MODEL RESULTS =====")
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

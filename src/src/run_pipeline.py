# ==========================================
# SYN Flood DDoS Detection – Master Pipeline
# ==========================================

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("\n===== STEP 1: LOAD DATA =====")
data = pd.read_csv("data/cic-ddos2019-final.csv")
X = data.drop("Label", axis=1)
y = data["Label"]

# Train-test split already exists logically
train_df = pd.read_csv("data/train_corrected.csv")
test_df = pd.read_csv("data/test.csv")

X_train = train_df.drop("Label", axis=1)
y_train = train_df["Label"]
X_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"]

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ------------------------------------------
print("\n===== STEP 2: BASELINE MODELS =====")

baseline_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

# ------------------------------------------
print("\n===== STEP 3: ADVANCED MODEL =====")

advanced_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

advanced_model.fit(X_train, y_train)
adv_preds = advanced_model.predict(X_test)

print("Advanced Model Results:")
print(confusion_matrix(y_test, adv_preds))
print(classification_report(y_test, adv_preds))

# ------------------------------------------
print("\n===== STEP 4: HYPERPARAMETER TUNING =====")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)
tuned_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

joblib.dump(tuned_model, "models/ddos_rf_tuned_model.pkl")

# ------------------------------------------
print("\n===== STEP 5: MODEL INTERPRETATION =====")

importance = tuned_model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print(feature_importance.head(10))

# ------------------------------------------
print("\n===== STEP 6: STATISTICAL VALIDATION =====")

baseline_cv = cross_val_score(
    LogisticRegression(max_iter=1000),
    X, y, cv=3, scoring="f1"
)

tuned_cv = cross_val_score(
    tuned_model,
    X, y, cv=3, scoring="f1"
)

print("Baseline Mean F1:", baseline_cv.mean())
print("Tuned Mean F1   :", tuned_cv.mean())

# ------------------------------------------
print("\n===== STEP 7: BUSINESS OPTIMIZATION =====")

proba = tuned_model.predict_proba(X_test)[:, 1]

pred_05 = (proba >= 0.5).astype(int)
pred_03 = (proba >= 0.3).astype(int)

print("Threshold 0.5 Confusion Matrix:")
print(confusion_matrix(y_test, pred_05))

print("Threshold 0.3 Confusion Matrix:")
print(confusion_matrix(y_test, pred_03))

# ------------------------------------------
print("\n===== STEP 8: FINAL MODEL SELECTION =====")

joblib.dump(tuned_model, "models/final_ddos_model.pkl")

print("Final Model Selected: Tuned Random Forest")
print("Final model saved as models/final_ddos_model.pkl")

print("\n===== PIPELINE COMPLETED SUCCESSFULLY =====")

# ------------------------------------------

print("\n===== REAL-TIME DDoS DETECTION SYSTEM =====")

# Load trained model ONCE
model = joblib.load("models/final_ddos_model.pkl")
print("Model loaded successfully.\n")

THRESHOLD = 0.4

while True:
    print("Enter network traffic values:")

    flow_duration = float(input("Flow Duration: "))
    total_fwd_packets = float(input("Total Fwd Packets: "))
    total_bwd_packets = float(input("Total Backward Packets: "))
    flow_bytes_s = float(input("Flow Bytes/s: "))
    flow_packets_s = float(input("Flow Packets/s: "))
    packet_length_mean = float(input("Packet Length Mean: "))
    packet_length_std = float(input("Packet Length Std: "))
    syn_flag_count = float(input("SYN Flag Count: "))
    ack_flag_count = float(input("ACK Flag Count: "))
    fwd_packet_len_mean = float(input("Fwd Packet Length Mean: "))
    bwd_packet_len_mean = float(input("Bwd Packet Length Mean: "))

    # Create input DataFrame (same order as training features)
    input_data = pd.DataFrame([[
        flow_duration,
        total_fwd_packets,
        total_bwd_packets,
        flow_bytes_s,
        flow_packets_s,
        packet_length_mean,
        packet_length_std,
        syn_flag_count,
        ack_flag_count,
        fwd_packet_len_mean,
        bwd_packet_len_mean
    ]], columns=[
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Packet Length Mean",
        "Packet Length Std",
        "SYN Flag Count",
        "ACK Flag Count",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean"
    ])

    # Predict threat score
    threat_score = model.predict_proba(input_data)[0][1]

    # Decision using threshold
    if threat_score >= THRESHOLD:
        decision = "SYN FLOOD ATTACK"
    else:
        decision = "BENIGN NETWORK"

    # Output
    print("\n===== DETECTION RESULT =====")
    print("Threat Score :", round(threat_score, 2))
    print("Decision     :", decision)
    print("-----------------------------")

    # Ask user to continue or exit
    choice = input("Do you want to test another input? (yes/no): ").lower()
    if choice != "yes":
        print("Detection system stopped.")
        break

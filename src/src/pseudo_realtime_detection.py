import joblib
import time
import pandas as pd

# =============================
# CONFIG
# =============================
MODEL_PATH = "models/ddos_random_forest_model.pkl"
THRESHOLD = 0.5

FEATURES = [
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
]

# =============================
# LOAD MODEL
# =============================
model = joblib.load(MODEL_PATH)

print("\n===== REAL-TIME DDoS DETECTION SYSTEM =====")
print("Model loaded successfully...\n")

# =============================
# REALISTIC STREAM (MATCHES TRAINING DATA)
# =============================
stream_data = [
    # -------- BENIGN --------
    [9000, 40, 35, 1200, 8, 60, 10, 1, 35, 70, 65],
    [10000, 55, 45, 1500, 9, 62, 11, 0, 40, 72, 68],

    # -------- SYN FLOOD ATTACK --------
    [500, 3000, 0, 90000, 120, 38, 2, 2500, 0, 36, 8],
    [400, 3500, 0, 110000, 150, 35, 1, 3200, 0, 34, 6],

    # -------- BENIGN --------
    [8500, 60, 50, 1800, 10, 65, 12, 0, 45, 75, 70],

    # -------- SYN FLOOD ATTACK --------
    [300, 4000, 0, 150000, 200, 32, 1, 4000, 0, 30, 5],
]

# =============================
# LOOP (PSEUDO REAL-TIME)
# =============================
log_id = 1

for values in stream_data:
    print(f"\n--- Log Entry {log_id} ---")

    df_input = pd.DataFrame([values], columns=FEATURES)

    attack_prob = model.predict_proba(df_input)[0][1]

    decision = (
        "SYN FLOOD ATTACK"
        if attack_prob >= THRESHOLD
        else "BENIGN NETWORK"
    )

    print(f"Threat Score : {round(attack_prob, 2)}")
    print(f"Decision     : {decision}")

    log_id += 1
    time.sleep(1)

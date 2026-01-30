import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/ddos_random_forest_model.pkl")

# ---- MANUAL INPUT SECTION ----
print("Enter Network Traffic Details:")

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

# Create DataFrame (order MUST match training data)
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

# Prediction
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

result = "SYN Attack" if prediction == 1 else "Benign"

print("\n===== PREDICTION RESULT =====")
print("Predicted Label :", result)
print("Threat Score    :", round(probability, 2))

import pandas as pd

# Load clean dataset
data = pd.read_csv(r"data\cic-ddos2019-final.csv")

print("===== ORIGINAL COLUMNS =====")
print(data.columns)

# ----------------------------
# Feature Engineering (CORRECTED)
# ----------------------------

# Feature 1: Packet rate (already exists but we recompute clearly)
data["packet_rate"] = data["Flow Packets/s"]

# Feature 2: Byte rate (already exists but recomputed)
data["byte_rate"] = data["Flow Bytes/s"]

# Feature 3: Average packet size
data["avg_packet_size"] = (
    data["Flow Bytes/s"] / (data["Flow Packets/s"] + 1)
)

print("\n===== NEW FEATURES CREATED (SAMPLE) =====")
print(data[["packet_rate", "byte_rate", "avg_packet_size"]].head())

print("\nTotal columns AFTER feature engineering:", data.shape[1])
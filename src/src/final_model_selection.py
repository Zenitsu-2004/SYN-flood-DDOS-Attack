import joblib

# Load tuned model
final_model = joblib.load("models/ddos_rf_tuned_model.pkl")

# Save as final deployment model
joblib.dump(final_model, "models/final_ddos_model.pkl")

print("===== FINAL MODEL SELECTION =====")
print("Selected Model: Tuned Random Forest")
print("Reason: Best F1-score, zero false negatives after threshold optimization")
print("Final model saved as: models/final_ddos_model.pkl")

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix

# ----------------------------------
# Create results folder
# ----------------------------------
os.makedirs("results", exist_ok=True)

# ----------------------------------
# Load train & test datasets
# ----------------------------------
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")

X_train = train_df.drop("Label", axis=1)
y_train = train_df["Label"]

X_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"]

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nTrain label distribution:")
print(y_train.value_counts())
print("\nTest label distribution:")
print(y_test.value_counts())

# ----------------------------------
# 1️⃣ CLASS DISTRIBUTION PLOT
# ----------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x=y_train)
plt.title("Class Distribution (Train Data)")
plt.xlabel("Label (0 = Benign, 1 = Attack)")
plt.ylabel("Count")
plt.savefig("results/train_class_distribution.png")
plt.close()

# ----------------------------------
# 2️⃣ FEATURE DISTRIBUTION (Histogram)
# ----------------------------------
sample_feature = X_train.columns[0]

plt.figure(figsize=(6,4))
sns.histplot(X_train[sample_feature], bins=30, kde=True)
plt.title(f"Feature Distribution: {sample_feature}")
plt.savefig("results/feature_distribution.png")
plt.close()

# ----------------------------------
# 3️⃣ CORRELATION HEATMAP (Top Features)
# ----------------------------------
corr = X_train.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig("results/correlation_heatmap.png")
plt.close()

# ----------------------------------
# Load trained model
# ----------------------------------
model = joblib.load("models/ddos_random_forest_model.pkl")

# ----------------------------------
# 4️⃣ CONFUSION MATRIX (Test Data)
# ----------------------------------
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Data)")
plt.savefig("results/confusion_matrix.png")
plt.close()

# ----------------------------------
# 5️⃣ FEATURE IMPORTANCE
# ----------------------------------
importances = model.feature_importances_
features = X_train.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(8,5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.title("Top 10 Important Features")
plt.savefig("results/feature_importance.png")
plt.close()

print("\n✅ EDA & visualization completed successfully")
print("📁 Charts saved in the 'results/' folder")

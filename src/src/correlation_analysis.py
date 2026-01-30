import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load clean dataset
data = pd.read_csv(r"data\cic-ddos2019-final.csv")

# Drop label column (not used for correlation)
features = data.drop("Label", axis=1)

# Compute correlation matrix
corr_matrix = features.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

# Save for demo proof
plt.savefig("reports/correlation_heatmap.png")
plt.show()

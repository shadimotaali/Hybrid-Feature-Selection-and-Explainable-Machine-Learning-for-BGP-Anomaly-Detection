import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load evaluation results
df = pd.read_csv("model_results/hybrid_models_evaluation.csv")

# Ensure output directory exists
os.makedirs("plots", exist_ok=True)

# Metrics to plot
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

# Create a 2x2 grid of bar plots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    ax = axs[i]
    sns.barplot(
        data=df,
        x="Dataset",
        y=metric,
        hue="Model",
        ax=ax
    )
    ax.set_title(f"{metric} by Dataset and Model")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric)
    ax.legend(title="Model")
    ax.grid(axis="y", linestyle='--', alpha=0.7)

plt.tight_layout()
plot_path = "plots/hybrid_models_evaluation_summary.png"
plt.savefig(plot_path)
plt.close()

print(f"✅ Combined evaluation plot saved as '{plot_path}'")

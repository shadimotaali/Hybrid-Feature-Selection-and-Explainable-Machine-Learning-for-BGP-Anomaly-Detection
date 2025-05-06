
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths to class directories
class_dirs = {
    "Class_1": "feature_selection_Class_1",
    "Class_2": "feature_selection_Class_2",
    "Class_3": "feature_selection_Class_3"
}

# Count how many times each feature appears across all algorithms and classes
feature_counts = {}

for class_name, class_dir in class_dirs.items():
    for file in os.listdir(class_dir):
        if file.endswith("_top_48.csv"):
            algo = file.replace("_top_48.csv", "")
            df = pd.read_csv(os.path.join(class_dir, file))
            for feature in df["Feature"]:
                key = (feature, class_name, algo)
                feature_counts[key] = feature_counts.get(key, 0) + 1

# Convert to DataFrame
records = [
    {"Feature": feat, "Class": cls, "Algorithm": algo, "Count": count}
    for (feat, cls, algo), count in feature_counts.items()
]
df_freq = pd.DataFrame(records)

# Pivot to create a heatmap: index=Feature, columns=Class_Algorithm, values=Count
df_freq["Class_Algo"] = df_freq["Class"] + " - " + df_freq["Algorithm"]
pivot_df = df_freq.pivot_table(index="Feature", columns="Class_Algo", values="Count", fill_value=0)

# Filter to most frequent features
feature_totals = pivot_df.sum(axis=1).sort_values(ascending=False)
top_features = feature_totals.head(30).index
pivot_top = pivot_df.loc[top_features]

# Plot heatmap
plt.figure(figsize=(18, 12))
sns.heatmap(pivot_top, cmap="YlGnBu", annot=True, fmt="d", linewidths=.5)
plt.title("Feature Occurrence Frequency Across All Classes and Algorithms (Top 30 Features)")
plt.tight_layout()
os.makedirs("feature_summary_plots", exist_ok=True)
plt.savefig("feature_summary_plots/feature_frequency_heatmap.png")
plt.close()

print("✅ Feature frequency heatmap saved in: feature_summary_plots/feature_frequency_heatmap.png")

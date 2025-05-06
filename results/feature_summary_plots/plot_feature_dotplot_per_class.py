
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Directories for each class
class_dirs = {
    "Class_1": "feature_selection_Class_1",
    "Class_2": "feature_selection_Class_2",
    "Class_3": "feature_selection_Class_3"
}

# Gather all feature importances
records = []

for class_label, path in class_dirs.items():
    for file in os.listdir(path):
        if file.endswith("_top_48.csv"):
            algo = file.replace("_top_48.csv", "")
            df = pd.read_csv(os.path.join(path, file))
            for i, row in df.head(20).iterrows():  # limit to top 20 per algo
                records.append({
                    "Feature": row["Feature"],
                    "Score": row["Score"],
                    "Algorithm": algo,
                    "Class": class_label
                })

# Create DataFrame
df = pd.DataFrame(records)

# Output directory
output_dir = "feature_summary_plots"
os.makedirs(output_dir, exist_ok=True)

# Dot plot per class
for class_label in df["Class"].unique():
    class_df = df[df["Class"] == class_label]
    plt.figure(figsize=(14, 10))
    sns.stripplot(data=class_df, y="Feature", x="Score", hue="Algorithm", dodge=True, jitter=True, size=7)
    plt.title(f"Top 20 Feature Importances per Algorithm - {class_label}")
    plt.tight_layout()
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0), frameon=True)
    plt.savefig(os.path.join(output_dir, f"{class_label}_dotplot_feature_importance.png"))
    plt.close()

print("✅ Dot plots for top 20 features saved in:", output_dir)

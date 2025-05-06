
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Class folders and names
class_dirs = {
    "Class_1": "feature_selection_Class_1",
    "Class_2": "feature_selection_Class_2",
    "Class_3": "feature_selection_Class_3"
}

output_dir = "feature_summary_plots"
os.makedirs(output_dir, exist_ok=True)

for class_name, path in class_dirs.items():
    all_scores = {}

    for file in os.listdir(path):
        if file.endswith("_top_48.csv"):
            algo = file.replace("_top_48.csv", "")
            df = pd.read_csv(os.path.join(path, file))
            for i, row in df.iterrows():
                feat = row["Feature"]
                if feat not in all_scores:
                    all_scores[feat] = {}
                all_scores[feat][algo] = row["Score"]

    # Convert to DataFrame and fill missing scores with 0
    score_df = pd.DataFrame.from_dict(all_scores, orient="index").fillna(0)

    # Sort features by the highest average importance
    score_df["mean_score"] = score_df.mean(axis=1)
    score_df = score_df.sort_values("mean_score", ascending=False).drop(columns="mean_score")

    # Plot heatmap
    plt.figure(figsize=(14, max(10, int(len(score_df) * 0.4))))
    sns.heatmap(score_df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5)
    plt.title(f"Feature Importance Comparison - {class_name} (Top 48 Features)")
    plt.xlabel("Algorithm")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{class_name}_feature_importance_comparison.png"))
    plt.close()

print("✅ Comparison heatmaps saved in:", output_dir)


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Class directories
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
        if file.endswith("_top_48.csv") and not ("Lasso_Regression" in file or "RFE" in file):
            algo = file.replace("_top_48.csv", "")
            df = pd.read_csv(os.path.join(path, file))
            for i, row in df.iterrows():
                feat = row["Feature"]
                if feat not in all_scores:
                    all_scores[feat] = {}
                all_scores[feat][algo] = row["Score"]

    # Create raw scores DataFrame
    raw_df = pd.DataFrame.from_dict(all_scores, orient="index").fillna(0)

    # Normalize scores for heatmap color (but annotate raw values)
    scaler = MinMaxScaler()
    norm_df = pd.DataFrame(scaler.fit_transform(raw_df), columns=raw_df.columns, index=raw_df.index)

    # Sort features by average raw importance
    raw_df["mean_score"] = raw_df.mean(axis=1)
    raw_df = raw_df.sort_values("mean_score", ascending=False).drop(columns="mean_score")
    norm_df = norm_df.loc[raw_df.index]  # keep same row order

    # Plot
    plt.figure(figsize=(14, max(10, int(len(raw_df) * 0.4))))
    sns.heatmap(norm_df, annot=raw_df, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
    plt.title(f"Normalized Feature Importance Comparison - {class_name} (Top 48 Features)")
    plt.xlabel("Algorithm")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{class_name}_normalized_feature_importance_comparison.png"))
    plt.close()

print("✅ Normalized comparison heatmaps saved in:", output_dir)

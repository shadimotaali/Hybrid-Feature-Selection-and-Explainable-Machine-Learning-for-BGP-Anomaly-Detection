
import os
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

# Valid algorithms to include (excluding Lasso and RFE)
valid_algorithms = [
    "ANOVA_F_score", "Mutual_Information", "Random_Forest", "XGBoost", "Gini_Decision_Tree"
]

# Class directories
class_dirs = {
    "Class_1": "feature_selection_Class_1",
    "Class_2": "feature_selection_Class_2",
    "Class_3": "feature_selection_Class_3"
}

# Store top-10 presence
feature_presence = defaultdict(lambda: defaultdict(int))

# Build presence matrix
for class_name, path in class_dirs.items():
    for file in os.listdir(path):
        if file.endswith("_top_48.csv"):
            algo = file.replace("_top_48.csv", "")
            if algo not in valid_algorithms:
                continue
            df = pd.read_csv(os.path.join(path, file))
            top_features = df.head(10)["Feature"]
            for feature in top_features:
                col = f"{class_name} | {algo}"
                feature_presence[feature][col] = 1

# Create DataFrame
presence_df = pd.DataFrame.from_dict(feature_presence, orient="index").fillna(0).astype(int)

# Add total count column
presence_df["Total_Top10_Appearances"] = presence_df.sum(axis=1)

# Sort and keep top 30 by total appearances
top_features_df = presence_df.sort_values("Total_Top10_Appearances", ascending=False).head(30)

# Output
output_dir = "feature_summary_plots"
os.makedirs(output_dir, exist_ok=True)

top_features_df.to_csv(os.path.join(output_dir, "top10_feature_consistency.csv"))

# Heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(top_features_df.drop(columns="Total_Top10_Appearances"), annot=True, cmap="crest", cbar=False, linewidths=0.5)
plt.title("Top 30 Features Consistently in Top-10 Across Classes and Algorithms")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top10_feature_consistency_heatmap.png"))
plt.close()

print("✅ Top-10 feature consistency matrix saved and plotted.")

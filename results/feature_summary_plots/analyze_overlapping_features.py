
import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to feature selection results
class_dirs = {
    "Class_1": "feature_selection_Class_1",
    "Class_2": "feature_selection_Class_2",
    "Class_3": "feature_selection_Class_3"
}

# Count feature occurrences across all classes and algorithms
feature_occurrence = defaultdict(int)

# Track per-feature appearances
feature_details = []

for class_name, path in class_dirs.items():
    for file in os.listdir(path):
        if file.endswith("_top_48.csv"):
            algo = file.replace("_top_48.csv", "")
            df = pd.read_csv(os.path.join(path, file))
            for feature in df["Feature"]:
                feature_occurrence[feature] += 1
                feature_details.append({
                    "Feature": feature,
                    "Class": class_name,
                    "Algorithm": algo
                })

# Convert to DataFrame
df_counts = pd.DataFrame.from_dict(feature_occurrence, orient='index', columns=['Count'])
df_counts = df_counts.sort_values(by="Count", ascending=False)

# Save as CSV
output_dir = "feature_summary_plots"
os.makedirs(output_dir, exist_ok=True)
df_counts.to_csv(os.path.join(output_dir, "top_overlapping_features.csv"))

# Plot top 30 overlapping features
top_n = 30
plt.figure(figsize=(14, 10))
sns.barplot(y=df_counts.head(top_n).index, x=df_counts.head(top_n)["Count"], palette="crest")
plt.title("Top Overlapping Features Across All Classes and Algorithms")
plt.xlabel("Occurrence Count")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_overlapping_features.png"))
plt.close()

print("✅ Overlapping features analysis saved in:", output_dir)

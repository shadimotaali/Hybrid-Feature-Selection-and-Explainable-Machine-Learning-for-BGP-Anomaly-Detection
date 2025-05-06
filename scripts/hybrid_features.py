import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# --- Configurable threshold ---
THRESHOLD_PERCENT = 75  # You can change this to 50, 60, 70, etc.

# --- Step 1: Load features from general feature_selection_results (full dataset) ---
general_features = []
for method in ["ANOVA_F_score.csv", "Mutual_Information.csv", "Random_Forest_Importance.csv", "XGBoost_Importance.csv"]:
    path = os.path.join("feature_selection_results", method)
    if os.path.exists(path):
        features = pd.read_csv(path)["Feature"].tolist()[:30]
        general_features.append(set(features))

# --- Step 2: Load features from class-specific feature_selection_Class_X folders ---
class_dirs = ["feature_selection_Class_1", "feature_selection_Class_2", "feature_selection_Class_3"]
class_features = []

for class_dir in class_dirs:
    for method in ["ANOVA_F_score.csv", "Mutual_Information.csv", "Random_Forest_Importance.csv", "XGBoost_Importance.csv"]:
        path = os.path.join(class_dir, method)
        if os.path.exists(path):
            features = pd.read_csv(path)["Feature"].tolist()[:30]
            class_features.append(set(features))

# --- Step 3: Merge all lists ---
all_feature_sets = general_features + class_features

# Total number of sources
total_lists = len(all_feature_sets)

# --- Step 4: Count occurrences ---
feature_counter = Counter()
for feature_set in all_feature_sets:
    feature_counter.update(feature_set)

# --- Step 5: Select features appearing in >= threshold% ---
min_required = int((THRESHOLD_PERCENT / 100) * total_lists)

hybrid_features = [(feature, count) for feature, count in feature_counter.items() if count >= min_required]
hybrid_features = sorted(hybrid_features, key=lambda x: (-x[1], x[0]))

# Save to CSV
os.makedirs("feature_selection_results", exist_ok=True)
hybrid_df = pd.DataFrame(hybrid_features, columns=["Feature", "Frequency"])
hybrid_df.to_csv("feature_selection_results/Hybrid_top_features.csv", index=False)

print(f"\n✅ Saved {len(hybrid_features)} hybrid features (appearing in \u2265{THRESHOLD_PERCENT}% of lists) to 'feature_selection_results/Hybrid_top_features.csv'")

# --- Step 6: Plot feature frequencies ---
if len(hybrid_df) > 0:
    plt.figure(figsize=(12, 6))
    plt.barh(hybrid_df["Feature"], hybrid_df["Frequency"], color="royalblue")
    plt.xlabel("Frequency (Appearing in # Lists)")
    plt.title(f"Top Hybrid Features (Threshold {THRESHOLD_PERCENT}% - Min {min_required} Lists)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_selection_results/Hybrid_top_features_plot.png")
    plt.close()
    print("✅ Saved hybrid feature plot to 'feature_selection_results/Hybrid_top_features_plot.png'")
else:
    print("⚠️ No features met the threshold, no plot generated.")

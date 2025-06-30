import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Create output directory
os.makedirs("feature_ablation_results", exist_ok=True)

print("🔍 Comprehensive Feature Ablation Study: Impact of all hybrid features")
print("=" * 80)

# Load the hybrid dataset
print("\n📊 Loading dataset...")
data = pd.read_excel("Randomly_Balanced_Dataset - Hybrid.xlsx")

# Clean up the data
X = data.drop(columns=["target"])
y = data["target"].apply(lambda x: 0 if x == 0 else 1)  # Normalize target to binary

# Define all 25 hybrid features based on the plot
all_hybrid_features = [
    "dups",
    "edit_distance_avg",
    "edit_distance_dict_0",
    "edit_distance_dict_1",
    "imp_wd",
    "imp_wd_spath",
    "nlri_ann",
    "origin_0",
    "origin_2",
    "rare_ases_avg",
    "unique_as_path_max",
    "announcements",
    "as_path_max",
    "edit_distance_dict_2",
    "edit_distance_dict_4",
    "edit_distance_dict_6",
    "edit_distance_max",
    "edit_distance_unique_dict_0",
    "edit_distance_unique_dict_1",
    "flaps",
    "imp_wd_dpath",
    "nadas",
    "number_rare_ases",
    "origin_changes",
    "withdrawals"
]

# Verify features exist in dataset
valid_features = [f for f in all_hybrid_features if f in X.columns]
missing_features = [f for f in all_hybrid_features if f not in X.columns]

if missing_features:
    print(f"\n⚠️ Warning: The following features were not found in the dataset: {missing_features}")
    print(f"Proceeding with {len(valid_features)} valid features.")

# Calculate variance for all hybrid features
print("\n📈 Analyzing feature variance...")
feature_variance = X[valid_features].var()
feature_var_df = pd.DataFrame({
    "Feature": feature_variance.index,
    "Variance": feature_variance.values
}).sort_values("Variance")

# Save variance data
feature_var_df.to_csv("feature_ablation_results/feature_variances.csv", index=False)
print(f"✅ Saved feature variance data to feature_ablation_results/feature_variances.csv")

# Display the 5 lowest variance features
print("\nLowest variance features:")
print(feature_var_df.head(5).to_string(index=False))

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Results storage
results = []

# First, establish baseline with all hybrid features
X_baseline = X[valid_features]
X_train, X_test, y_train, y_test = train_test_split(X_baseline, y, test_size=0.2, stratify=y, random_state=42)

print("\n🔍 Establishing baseline performance with all hybrid features...")
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    results.append({
        "Model": model_name,
        "Feature_Removed": "None (Baseline)",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "F1_Change": 0.0,
        "F1_Change_Percent": 0.0
    })
    print(f"  {model_name} baseline F1: {f1:.4f}")

# Now test by removing one feature at a time
print("\n🧪 Evaluating impact of removing individual features...")
for feature_to_remove in valid_features:
    print(f"  Testing without feature: {feature_to_remove}")
    
    # Create feature subset without the current feature
    ablated_features = [f for f in valid_features if f != feature_to_remove]
    
    X_ablated = X[ablated_features]
    X_train, X_test, y_train, y_test = train_test_split(X_ablated, y, test_size=0.2, stratify=y, random_state=42)
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        # Find baseline F1 for this model
        baseline_f1 = next(item["F1_Score"] for item in results if item["Model"] == model_name and item["Feature_Removed"] == "None (Baseline)")
        f1_change = f1 - baseline_f1
        f1_change_pct = (f1_change / baseline_f1) * 100
        
        results.append({
            "Model": model_name,
            "Feature_Removed": feature_to_remove,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "F1_Change": f1_change,
            "F1_Change_Percent": f1_change_pct
        })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("feature_ablation_results/complete_feature_ablation_metrics.csv", index=False)
print(f"\n✅ Saved detailed metrics to feature_ablation_results/complete_feature_ablation_metrics.csv")

# Merge with variance data for analysis
merged_data = []
for idx, row in results_df[results_df["Feature_Removed"] != "None (Baseline)"].iterrows():
    feature = row["Feature_Removed"]
    variance = feature_var_df[feature_var_df["Feature"] == feature]["Variance"].values
    if len(variance) > 0:
        merged_data.append({
            "Feature": feature,
            "Model": row["Model"],
            "Variance": variance[0],
            "F1_Change": row["F1_Change"],
            "F1_Change_Percent": row["F1_Change_Percent"]
        })

merged_df = pd.DataFrame(merged_data)
merged_df.to_csv("feature_ablation_results/variance_impact_data.csv", index=False)

# Create plot of F1 score changes - Top 10 most impactful features
plt.figure(figsize=(14, 8))
top_impact_df = results_df[results_df["Feature_Removed"] != "None (Baseline)"].copy()
# Calculate absolute impact
top_impact_df["Abs_Impact"] = top_impact_df["F1_Change_Percent"].abs()
# Get the top 10 features with highest impact for each model
rf_top10 = top_impact_df[top_impact_df["Model"] == "Random Forest"].nlargest(10, "Abs_Impact")["Feature_Removed"].unique()
xgb_top10 = top_impact_df[top_impact_df["Model"] == "XGBoost"].nlargest(10, "Abs_Impact")["Feature_Removed"].unique()
# Combine top features from both models
top_features = list(set(rf_top10) | set(xgb_top10))
# Filter dataframe to include only top features
plot_df = top_impact_df[top_impact_df["Feature_Removed"].isin(top_features)]

# Create grouped bar chart
sns.barplot(x="Feature_Removed", y="F1_Change_Percent", hue="Model", data=plot_df)
plt.axhline(y=0, color='gray', linestyle='--')
plt.title("Impact of Feature Removal on F1 Score (Top Impact Features)", fontsize=14)
plt.xlabel("Removed Feature", fontsize=12)
plt.ylabel("F1 Score Change (%)", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("feature_ablation_results/top_features_impact.png")
print(f"✅ Saved top impact features visualization to feature_ablation_results/top_features_impact.png")

# Create heatmap of feature impact
# Pivot the data for the heatmap
pivot_df = results_df[results_df["Feature_Removed"] != "None (Baseline)"].pivot(
    index="Feature_Removed", columns="Model", values="F1_Change_Percent"
)

# Sort by average impact
pivot_df["Average_Impact"] = pivot_df.mean(axis=1)
pivot_df = pivot_df.sort_values("Average_Impact")
pivot_df = pivot_df.drop(columns=["Average_Impact"])

plt.figure(figsize=(10, 16))
sns.heatmap(pivot_df, cmap="RdBu_r", center=0, annot=True, fmt=".2f", 
            cbar_kws={"label": "F1 Score Change (%)"})
plt.title("Impact of Feature Removal on Model Performance", fontsize=14)
plt.tight_layout()
plt.savefig("feature_ablation_results/feature_impact_heatmap.png")
print(f"✅ Saved feature impact heatmap to feature_ablation_results/feature_impact_heatmap.png")

# Plot feature variance vs F1 impact
plt.figure(figsize=(12, 8))
sns.scatterplot(x="Variance", y="F1_Change", hue="Model", 
                style="Model", s=100, data=merged_df)

# Add feature labels to points with significant impact
threshold = merged_df["F1_Change"].abs().mean() * 1.5
for _, row in merged_df[merged_df["F1_Change"].abs() > threshold].iterrows():
    plt.annotate(row["Feature"], 
                (row["Variance"], row["F1_Change"]),
                xytext=(5, 5), textcoords="offset points")

plt.title("Feature Variance vs. F1 Score Impact When Removed", fontsize=14)
plt.xlabel("Feature Variance", fontsize=12)
plt.ylabel("F1 Score Change", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("feature_ablation_results/complete_variance_vs_impact.png")
print(f"✅ Saved variance-impact analysis to feature_ablation_results/complete_variance_vs_impact.png")

# Create a more detailed analysis of low-variance but high-impact features
low_var_threshold = feature_var_df["Variance"].median()  # Define low variance as below median
merged_df["Variance_Category"] = merged_df["Variance"].apply(
    lambda x: "Low Variance" if x < low_var_threshold else "High Variance"
)
merged_df["Impact_Category"] = merged_df["F1_Change"].apply(
    lambda x: "High Impact" if abs(x) > threshold else "Low Impact"
)

# Count features in each category
category_counts = merged_df.groupby(["Variance_Category", "Impact_Category", "Model"]).size().reset_index(name="Count")

# Bar plot of categories
plt.figure(figsize=(12, 6))
g = sns.catplot(
    data=category_counts,
    kind="bar",
    x="Variance_Category", y="Count", hue="Impact_Category", col="Model",
    height=5, aspect=1.2
)
g.set_axis_labels("Variance Category", "Number of Features")
g.set_titles("{col_name}")
plt.tight_layout()
plt.savefig("feature_ablation_results/variance_impact_categories.png")
print(f"✅ Saved variance-impact category analysis to feature_ablation_results/variance_impact_categories.png")

# Generate table of low-variance, high-impact features
low_var_high_impact = merged_df[
    (merged_df["Variance_Category"] == "Low Variance") & 
    (merged_df["Impact_Category"] == "High Impact")
].sort_values(by=["Model", "F1_Change"])

if not low_var_high_impact.empty:
    low_var_high_impact.to_csv("feature_ablation_results/low_variance_high_impact_features.csv", index=False)
    print(f"✅ Saved analysis of low-variance, high-impact features to feature_ablation_results/low_variance_high_impact_features.csv")

    # Create a more readable table visualization
    plt.figure(figsize=(12, max(6, len(low_var_high_impact) * 0.4)))
    plt.axis('off')
    
    # Format data for table
    table_data = []
    for _, row in low_var_high_impact.iterrows():
        table_data.append([
            row["Feature"],
            row["Model"],
            f"{row['Variance']:.2e}",
            f"{row['F1_Change_Percent']:.2f}%"
        ])
    
    headers = ["Feature", "Model", "Variance", "F1 Impact (%)"]
    table = plt.table(
        cellText=table_data, 
        colLabels=headers, 
        loc='center', 
        cellLoc='center',
        colWidths=[0.3, 0.2, 0.25, 0.25]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.title("Low Variance Features with High Impact on Model Performance", fontsize=14)
    plt.tight_layout()
    plt.savefig("feature_ablation_results/low_variance_high_impact_table.png")
    print(f"✅ Saved table of low-variance, high-impact features to feature_ablation_results/low_variance_high_impact_table.png")

print("\n✅ Comprehensive feature ablation analysis complete! All results saved to 'feature_ablation_results/' directory.")

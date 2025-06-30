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

print("🔍 Feature Ablation Study: Impact of removing low-variance features")
print("=" * 80)

# Load the hybrid dataset
print("\n📊 Loading dataset...")
data = pd.read_excel("Randomly_Balanced_Dataset - Hybrid.xlsx")

# Clean up the data
X = data.drop(columns=["target"])
y = data["target"].apply(lambda x: 0 if x == 0 else 1)  # Normalize target to binary

# Load the hybrid features list
hybrid_features = pd.read_csv("feature_selection_results/Hybrid_top_features.csv")["Feature"].tolist()
top_features = hybrid_features[:25]  # Take top 25 features

# Calculate variance for all hybrid features
print("\n📈 Analyzing feature variance...")
feature_variance = X[top_features].var()
feature_var_df = pd.DataFrame({
    "Feature": feature_variance.index,
    "Variance": feature_variance.values
}).sort_values("Variance")

# Display the 5 lowest variance features
print("\nLowest variance features:")
print(feature_var_df.head(5).to_string(index=False))

# Define features to ablate (remove one by one)
# Focus on low-variance but important features
features_to_ablate = [
    "rare_ases_avg",
    "edit_distance_avg",
    "rare_ases_max",
    "dups",          # High semantic importance
    "nlri_ann",      # High semantic importance 
    "imp_wd_spath"   # Important for path stability
]

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Results storage
results = []

# First, establish baseline with all top-25 hybrid features
X_baseline = X[top_features]
X_train, X_test, y_train, y_test = train_test_split(X_baseline, y, test_size=0.2, stratify=y, random_state=42)

print("\n🔍 Establishing baseline performance with all 25 hybrid features...")
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
        "F1_Change": 0.0
    })
    print(f"  {model_name} baseline F1: {f1:.4f}")

# Now test by removing one feature at a time
print("\n🧪 Evaluating impact of removing individual features...")
for feature_to_remove in features_to_ablate:
    print(f"  Testing without feature: {feature_to_remove}")
    
    # Create feature subset without the current feature
    ablated_features = [f for f in top_features if f != feature_to_remove]
    
    # Skip if feature not in baseline
    if len(ablated_features) == len(top_features):
        print(f"    Warning: Feature {feature_to_remove} not found in baseline features. Skipping.")
        continue
    
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
        print(f"    {model_name} F1: {f1:.4f} (Change: {f1_change_pct:.2f}%)")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("feature_ablation_results/feature_ablation_metrics.csv", index=False)
print(f"\n✅ Saved detailed metrics to feature_ablation_results/feature_ablation_metrics.csv")

# Create plot of F1 score changes
plt.figure(figsize=(12, 8))
plot_df = results_df[results_df["Feature_Removed"] != "None (Baseline)"].copy()

# Create grouped bar chart
sns.barplot(x="Feature_Removed", y="F1_Change_Percent", hue="Model", data=plot_df)
plt.axhline(y=0, color='gray', linestyle='--')
plt.title("Impact of Feature Removal on F1 Score", fontsize=14)
plt.xlabel("Removed Feature", fontsize=12)
plt.ylabel("F1 Score Change (%)", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("feature_ablation_results/feature_ablation_impact.png")
print(f"✅ Saved impact visualization to feature_ablation_results/feature_ablation_impact.png")

# Create a table showing the raw F1 scores
table_data = []
headers = ["Feature Removed", "Random Forest F1", "XGBoost F1", "RF Change (%)", "XGB Change (%)"]

for feature in ["None (Baseline)"] + features_to_ablate:
    rf_data = results_df[(results_df["Model"] == "Random Forest") & 
                      (results_df["Feature_Removed"] == feature)]
    xgb_data = results_df[(results_df["Model"] == "XGBoost") & 
                       (results_df["Feature_Removed"] == feature)]
    
    # Only add row if we have data for this feature
    if not rf_data.empty and not xgb_data.empty:
        rf_f1 = rf_data["F1_Score"].values[0]
        xgb_f1 = xgb_data["F1_Score"].values[0]
        
        rf_change = rf_data["F1_Change_Percent"].values[0] if "F1_Change_Percent" in rf_data.columns else 0
        xgb_change = xgb_data["F1_Change_Percent"].values[0] if "F1_Change_Percent" in xgb_data.columns else 0
        
        table_data.append([
            feature, 
            f"{rf_f1:.4f}", 
            f"{xgb_f1:.4f}",
            f"{rf_change:.2f}%" if feature != "None (Baseline)" else "0.00%",
            f"{xgb_change:.2f}%" if feature != "None (Baseline)" else "0.00%"
        ])

# Create and save the table as a figure
plt.figure(figsize=(12, 6))
plt.axis('off')
table = plt.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.title("F1 Score Comparison After Feature Removal", fontsize=14)
plt.tight_layout()
plt.savefig("feature_ablation_results/feature_ablation_table.png")
print(f"✅ Saved performance table to feature_ablation_results/feature_ablation_table.png")

# Plot feature variance vs F1 impact
print("\n📊 Analyzing relationship between feature variance and model impact...")
variance_impact = []

for feature in features_to_ablate:
    if feature in feature_var_df["Feature"].values:
        var_value = feature_var_df[feature_var_df["Feature"] == feature]["Variance"].values[0]
        
        # Get F1 impact for Random Forest
        rf_impact = results_df[(results_df["Model"] == "Random Forest") & 
                             (results_df["Feature_Removed"] == feature)]["F1_Change"].values
        
        # Get F1 impact for XGBoost
        xgb_impact = results_df[(results_df["Model"] == "XGBoost") & 
                              (results_df["Feature_Removed"] == feature)]["F1_Change"].values
        
        if len(rf_impact) > 0:
            variance_impact.append({
                "Feature": feature,
                "Variance": var_value,
                "F1_Impact": abs(rf_impact[0]),
                "Model": "Random Forest"
            })
        
        if len(xgb_impact) > 0:
            variance_impact.append({
                "Feature": feature,
                "Variance": var_value,
                "F1_Impact": abs(xgb_impact[0]),
                "Model": "XGBoost"
            })

variance_impact_df = pd.DataFrame(variance_impact)

if not variance_impact_df.empty:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Variance", y="F1_Impact", hue="Model", 
                   style="Model", s=100, data=variance_impact_df)
    
    # Add feature labels
    for _, row in variance_impact_df.iterrows():
        plt.annotate(row["Feature"], 
                    (row["Variance"], row["F1_Impact"]),
                    xytext=(5, 5), textcoords="offset points")
    
    plt.title("Feature Variance vs. F1 Score Impact When Removed", fontsize=14)
    plt.xlabel("Feature Variance", fontsize=12)
    plt.ylabel("Absolute F1 Score Impact", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("feature_ablation_results/variance_vs_impact.png")
    print(f"✅ Saved variance-impact analysis to feature_ablation_results/variance_vs_impact.png")

print("\n✅ Feature ablation analysis complete! All results saved to 'feature_ablation_results/' directory.")

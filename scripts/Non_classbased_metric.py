import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

print("Processing Randomly Balanced Dataset...")

# Load the dataset
data = pd.read_excel("Randomly_Balanced_Dataset.xlsx")

# Drop datetime/timestamp columns if any
datetime_cols = data.select_dtypes(include=["datetime64[ns]", "datetime64", "object"]).columns
data = data.drop(columns=[col for col in datetime_cols if "timestamp" in col.lower()])

# Normalize labels to binary
data["target"] = data["target"].apply(lambda x: 0 if x == 0 else 1)

X = data.drop(columns=["target"])
y = data["target"]

# Feature selection results directory
features_dir = "./feature_selection_results"

# Define feature selection methods and corresponding files
feature_methods = {
    "ANOVA_F_score": "ANOVA_F_score.csv",
    "Random_Forest_Importance": "Random_Forest_Importance.csv",
    "XGBoost_Importance": "XGBoost_Importance.csv",
    "Mutual_Information": "Mutual_Information.csv",
    "Lasso_Coefficient": "Lasso_Coefficient.csv",
    "RFE_Ranking": "RFE_Ranking.csv",
}

# Storage for results
results = []

# Define classifiers
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Process each feature selection method
for method_name, file_name in feature_methods.items():
    file_path = f"{features_dir}/{file_name}"
    
    # Skip if file doesn't exist
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Skipping...")
        continue
    
    print(f"Processing feature selection method: {method_name}")
    
    # Load feature selection results
    feature_df = pd.read_csv(file_path)
    
    # Standardize column names
    if 'Feature' not in feature_df.columns:
        # Find the feature name column (usually the first column)
        feature_col = feature_df.columns[0]
        feature_df.rename(columns={feature_col: 'Feature'}, inplace=True)
    
    # Get features - take top 48 if available
    top_features = feature_df['Feature'].tolist()
    features_to_use = top_features[:min(48, len(top_features))]
    
    # Ensure features exist in dataset
    valid_features = [f for f in features_to_use if f in X.columns]
    
    if len(valid_features) == 0:
        print(f"Warning: No valid features found for {method_name}. Skipping...")
        continue
    
    # Select features from dataset
    X_selected = X[valid_features]
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name} with {method_name} features...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results.append({
            "Feature Selection Method": method_name,
            "Model": model_name,
            "Features Used": len(valid_features),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("randomly_balanced_metrics_summary.csv", index=False)

# Create plots directory
os.makedirs("feature_selection_plots", exist_ok=True)

# Create plots for each metric
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
for metric in metrics:
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_results, x="Feature Selection Method", y=metric, hue="Model")
    plt.title(f"{metric} by Feature Selection Method (Randomly Balanced Dataset)")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"feature_selection_plots/{metric.lower().replace(' ', '_')}_comparison.png")
    plt.close()

# Create a heatmap of F1 scores
pivot_df = df_results.pivot_table(
    index="Feature Selection Method", 
    columns="Model", 
    values="F1 Score"
)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title("F1 Score by Feature Selection Method and Model")
plt.tight_layout()
plt.savefig("feature_selection_plots/f1_score_heatmap.png")
plt.close()

print("Analysis completed. Results saved to randomly_balanced_metrics_summary.csv")
print("Plots saved to feature_selection_plots directory")

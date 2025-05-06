
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load and clean dataset
df = pd.read_excel("Randomly_Balanced_Dataset.xlsx")
df = df.select_dtypes(include=[np.number])  # Ensure only numeric columns
df = df.dropna()  # Remove any rows with NaNs

# Define top 48 features
top_48_features = df.columns.drop("target").tolist()[:48]

feature_set_sizes = list(range(48, -1, -5))  # From 48 down to 0 by -5
results = []

y = df["target"]

# Loop through feature set sizes
for k in feature_set_sizes:
    if k == 0:
        continue  # skip zero features case
    selected_features = top_48_features[:k]
    X = df[selected_features].astype(np.float32)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds, average='weighted')
    rf_precision = precision_score(y_test, rf_preds, average='weighted', zero_division=0)
    rf_recall = recall_score(y_test, rf_preds, average='weighted', zero_division=0)

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_f1 = f1_score(y_test, xgb_preds, average='weighted')
    xgb_precision = precision_score(y_test, xgb_preds, average='weighted', zero_division=0)
    xgb_recall = recall_score(y_test, xgb_preds, average='weighted', zero_division=0)

    # Store results
    results.append({
        "Features": k, "Model": "Random Forest", "Accuracy": rf_acc,
        "Precision": rf_precision, "Recall": rf_recall, "F1 Score": rf_f1
    })
    results.append({
        "Features": k, "Model": "XGBoost", "Accuracy": xgb_acc,
        "Precision": xgb_precision, "Recall": xgb_recall, "F1 Score": xgb_f1
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv("compare_rf_xgb_by_features_full_metrics_down_to_0.csv", index=False)

# Plotting metrics
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.ravel()
for i, metric in enumerate(metrics):
    for model in results_df["Model"].unique():
        subset = results_df[results_df["Model"] == model]
        axs[i].plot(subset["Features"], subset[metric], marker='o', label=model)
    axs[i].set_title(f"{metric} vs Number of Features")
    axs[i].set_xlabel("Number of Features")
    axs[i].set_ylabel(metric)
    axs[i].grid(True)
    axs[i].legend()

plt.tight_layout()
plt.savefig("model_full_metrics_comparison_down_to_0.png")
print("✅ Full metric comparison down to 0 features completed and results saved.")
print(results_df)

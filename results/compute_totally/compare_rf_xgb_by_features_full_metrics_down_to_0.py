
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_excel("Randomly_Balanced_Dataset.xlsx")
X_full = df.drop(columns=["target"])
y = df["target"]

# Cross-validation config
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall': make_scorer(recall_score, average='weighted', zero_division=0),
    'f1': make_scorer(f1_score, average='weighted', zero_division=0)
}

# Algorithms and corresponding top feature file names
model_map = {
    "Random Forest": ("Random_Forest_Importance.csv", RandomForestClassifier(random_state=42)),
    "XGBoost": ("XGBoost_Importance.csv", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
}

# Feature sizes to evaluate
feature_sizes = list(range(48, 0, -5))

# Run and collect results
results = []

for model_name, (feature_file, model) in model_map.items():
    top_features = pd.read_csv(f"./feature_selection_results/{feature_file}")["Feature"].tolist()
    for k in feature_sizes:
        selected = top_features[:k]
        X_subset = X_full[selected]
        scores = cross_validate(model, X_subset, y, cv=cv, scoring=scoring)
        results.append({
            "Model": model_name,
            "Features": k,
            "Accuracy": scores['test_accuracy'].mean(),
            "Precision": scores['test_precision'].mean(),
            "Recall": scores['test_recall'].mean(),
            "F1 Score": scores['test_f1'].mean()
        })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("compare_rf_xgb_by_features_full_metrics_down_to_0.csv", index=False)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Plot metrics vs features
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
df = pd.read_csv("compare_rf_xgb_by_features_full_metrics_down_to_0.csv")

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    ax = axs[i]
    for model in df["Model"].unique():
        subset = df[df["Model"] == model]
        ax.plot(subset["Features"], subset[metric], marker="o", label=model)
    ax.set_title(f"{metric} vs Number of Features")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel(metric)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig("combined_model_metrics_vs_features.png")
plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
import os
import time

# Create output directories
os.makedirs("grid_search_results", exist_ok=True)
os.makedirs("grid_search_plots", exist_ok=True)

print("Starting enhanced grid search hyperparameter tuning...")

# Load the dataset
data = pd.read_excel("Randomly_Balanced_Dataset - Hybrid.xlsx")
X = data.drop(columns=["target"])
y = data["target"].apply(lambda x: 0 if x == 0 else 1)

# Load hybrid features
hybrid_features = pd.read_csv("feature_selection_results/Hybrid_top_features.csv")["Feature"].tolist()[:25]
X_selected = X[hybrid_features]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)

# F1 scorer for optimization
f1_scorer = make_scorer(f1_score, average='weighted')

# Function to evaluate models
def evaluate_model(model, name, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'Training Time (s)': train_time,
        'Inference Time (s)': inference_time,
        'Confusion Matrix': cm
    }

# Store results
results = []

# 1. Evaluate default models for baseline
print("Evaluating default models...")
rf_default = RandomForestClassifier(random_state=42)
xgb_default = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

rf_default_results = evaluate_model(rf_default, "RF Default", X_train, X_test, y_train, y_test)
xgb_default_results = evaluate_model(xgb_default, "XGB Default", X_train, X_test, y_train, y_test)

results.append(rf_default_results)
results.append(xgb_default_results)

# 2. Define parameter grids for Random Forest
print("\nPerforming grid search for Random Forest...")
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)
rf_best_params = rf_grid.best_params_
rf_best_score = rf_grid.best_score_

print(f"Best Random Forest parameters: {rf_best_params}")
print(f"Best Random Forest CV score: {rf_best_score}")

# 3. Define parameter grid for XGBoost
print("\nPerforming grid search for XGBoost...")
xgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

xgb_grid = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_grid=xgb_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train, y_train)
xgb_best_params = xgb_grid.best_params_
xgb_best_score = xgb_grid.best_score_

print(f"Best XGBoost parameters: {xgb_best_params}")
print(f"Best XGBoost CV score: {xgb_best_score}")

# 4. Evaluate tuned models
print("\nEvaluating tuned models...")
rf_tuned = RandomForestClassifier(random_state=42, **rf_best_params)
xgb_tuned = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **xgb_best_params)

rf_tuned_results = evaluate_model(rf_tuned, "RF Tuned", X_train, X_test, y_train, y_test)
xgb_tuned_results = evaluate_model(xgb_tuned, "XGB Tuned", X_train, X_test, y_train, y_test)

results.append(rf_tuned_results)
results.append(xgb_tuned_results)

# 5. Add a simple ensemble (voting) approach
print("\nEvaluating ensemble approach...")
rf_probs = rf_tuned.predict_proba(X_test)[:, 1]
xgb_probs = xgb_tuned.predict_proba(X_test)[:, 1]
ensemble_preds = (rf_probs + xgb_probs) / 2
ensemble_preds_binary = (ensemble_preds > 0.5).astype(int)

acc = accuracy_score(y_test, ensemble_preds_binary)
prec = precision_score(y_test, ensemble_preds_binary, average="weighted", zero_division=0)
rec = recall_score(y_test, ensemble_preds_binary, average="weighted", zero_division=0)
f1 = f1_score(y_test, ensemble_preds_binary, average="weighted", zero_division=0)
cm = confusion_matrix(y_test, ensemble_preds_binary)

ensemble_results = {
    'Model': "Ensemble (RF+XGB)",
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1 Score': f1,
    'Training Time (s)': None,  # Not applicable
    'Inference Time (s)': None,  # Not applicable
    'Confusion Matrix': cm
}

results.append(ensemble_results)

# 6. Save results to CSV
print("\nSaving results...")
results_for_csv = []
for result in results:
    result_copy = result.copy()
    if 'Confusion Matrix' in result_copy:
        del result_copy['Confusion Matrix']
    results_for_csv.append(result_copy)

results_df = pd.DataFrame(results_for_csv)
results_df.to_csv("grid_search_results/grid_search_results.csv", index=False)

# 7. Save best parameters
params_df = pd.DataFrame({
    'RF_Best_Params': [rf_best_params],
    'XGB_Best_Params': [xgb_best_params],
    'RF_Best_CV_Score': [rf_best_score],
    'XGB_Best_CV_Score': [xgb_best_score]
})
params_df.to_csv("grid_search_results/best_parameters.csv", index=False)

# 8. Plot confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Confusion Matrices')

models = [
    ("RF Default", rf_default_results['Confusion Matrix'], 0, 0),
    ("XGB Default", xgb_default_results['Confusion Matrix'], 0, 1),
    ("RF Tuned", rf_tuned_results['Confusion Matrix'], 1, 0),
    ("XGB Tuned", xgb_tuned_results['Confusion Matrix'], 1, 1)
]

for model_name, cm, row, col in models:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[row, col])
    axes[row, col].set_title(model_name)
    axes[row, col].set_xlabel("Predicted")
    axes[row, col].set_ylabel("True")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("grid_search_plots/confusion_matrices.png")

# 9. Plot performance metrics
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='Model', y='F1 Score')
plt.title('F1 Score Comparison')
plt.ylim(0.85, 1.0)  # Adjust as needed
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("grid_search_plots/f1_score_comparison.png")

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='Model', y='Accuracy')
plt.title('Accuracy Comparison')
plt.ylim(0.85, 1.0)  # Adjust as needed
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("grid_search_plots/accuracy_comparison.png")

# 10. Print recommendations
print("\n====== RECOMMENDATIONS ======")
best_model_idx = results_df['F1 Score'].idxmax()
best_model = results_df.iloc[best_model_idx]
print(f"Best model: {best_model['Model']} with F1 Score: {best_model['F1 Score']:.4f}")

if best_model['Model'] == "RF Tuned":
    print(f"Recommended RF parameters: {rf_best_params}")
elif best_model['Model'] == "XGB Tuned":
    print(f"Recommended XGB parameters: {xgb_best_params}")
elif best_model['Model'] == "Ensemble (RF+XGB)":
    print("Recommended approach: Use ensemble method combining RF and XGB")
    print(f"RF parameters: {rf_best_params}")
    print(f"XGB parameters: {xgb_best_params}")
else:
    print(f"Recommended approach: Use default parameters as they perform best")

print("\nGrid search hyperparameter tuning completed!")
print("Results saved to 'grid_search_results/'")
print("Plots saved to 'grid_search_plots/'")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import os
import time

# Create output directories
os.makedirs("advanced_tuning_results", exist_ok=True)
os.makedirs("advanced_tuning_plots", exist_ok=True)

print("Starting advanced hyperparameter tuning experiment...")

# Load the dataset
data = pd.read_excel("Randomly_Balanced_Dataset - Hybrid.xlsx")
X = data.drop(columns=["target"])
y = data["target"].apply(lambda x: 0 if x == 0 else 1)

# Load hybrid features
hybrid_features = pd.read_csv("feature_selection_results/Hybrid_top_features.csv")["Feature"].tolist()[:25]
X_selected = X[hybrid_features]

# Split the data - use a different random state than previous experiments
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=123
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
    
    return {
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'Training Time (s)': train_time,
        'Inference Time (s)': inference_time
    }

# Store results
results = []

# Step 1: Evaluate default models for baseline
print("Evaluating default models...")
rf_default = RandomForestClassifier(random_state=42)
xgb_default = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

results.append(evaluate_model(rf_default, "RF Default", X_train, X_test, y_train, y_test))
results.append(evaluate_model(xgb_default, "XGB Default", X_train, X_test, y_train, y_test))

print("Default model evaluation complete.")

# Step 2: Two-Stage Tuning for Random Forest
# Stage 1: Broad RandomizedSearchCV
print("\nPerforming Stage 1 (Broad) randomized search for Random Forest...")

# Broader parameter distributions
rf_param_distributions = {
    'n_estimators': np.arange(50, 500, 50),
    'max_depth': [None] + list(np.arange(10, 100, 20)),
    'min_samples_split': np.arange(2, 20, 4),
    'min_samples_leaf': np.arange(1, 10, 2),
    'max_features': ['sqrt', 'log2', None]
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=rf_param_distributions,
    n_iter=25,  # Number of parameter settings sampled
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

rf_random.fit(X_train, y_train)
print(f"Stage 1 Best RF parameters: {rf_random.best_params_}")
print(f"Stage 1 Best RF score: {rf_random.best_score_}")

# Stage 2: Focused GridSearchCV around the best parameters from Stage 1
print("\nPerforming Stage 2 (Focused) grid search for Random Forest...")
best_n_estimators = rf_random.best_params_['n_estimators']
best_max_depth = rf_random.best_params_['max_depth']
best_min_samples_split = rf_random.best_params_['min_samples_split']
best_min_samples_leaf = rf_random.best_params_['min_samples_leaf']
best_max_features = rf_random.best_params_['max_features']

# Create more focused search space around best parameters
rf_param_grid_focused = {
    'n_estimators': [max(50, best_n_estimators - 50), best_n_estimators, min(500, best_n_estimators + 50)],
    'max_depth': [None] if best_max_depth is None else [max(10, best_max_depth - 10), best_max_depth, min(100, best_max_depth + 10)],
    'min_samples_split': [max(2, best_min_samples_split - 2), best_min_samples_split, min(20, best_min_samples_split + 2)],
    'min_samples_leaf': [max(1, best_min_samples_leaf - 1), best_min_samples_leaf, min(10, best_min_samples_leaf + 1)],
    'max_features': [best_max_features]
}

rf_grid_focused = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid_focused,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=1
)

rf_grid_focused.fit(X_train, y_train)
print(f"Stage 2 Best RF parameters: {rf_grid_focused.best_params_}")
print(f"Stage 2 Best RF score: {rf_grid_focused.best_score_}")

# Step 3: Two-Stage Tuning for XGBoost
# Stage 1: Broad RandomizedSearchCV
print("\nPerforming Stage 1 (Broad) randomized search for XGBoost...")
xgb_param_distributions = {
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    'max_depth': np.arange(3, 12, 2),
    'min_child_weight': np.arange(1, 10, 2),
    'subsample': np.arange(0.5, 1.05, 0.1),
    'colsample_bytree': np.arange(0.5, 1.05, 0.1),
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'n_estimators': np.arange(50, 500, 50)
}

xgb_random = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_distributions=xgb_param_distributions,
    n_iter=25,  # Number of parameter settings sampled
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

xgb_random.fit(X_train, y_train)
print(f"Stage 1 Best XGB parameters: {xgb_random.best_params_}")
print(f"Stage 1 Best XGB score: {xgb_random.best_score_}")

# Stage 2: Focused GridSearchCV around the best parameters from Stage 1
print("\nPerforming Stage 2 (Focused) grid search for XGBoost...")
best_lr = xgb_random.best_params_['learning_rate']
best_max_depth = xgb_random.best_params_['max_depth']
best_min_child_weight = xgb_random.best_params_['min_child_weight']
best_subsample = xgb_random.best_params_['subsample']
best_colsample_bytree = xgb_random.best_params_['colsample_bytree']
best_gamma = xgb_random.best_params_['gamma']
best_n_estimators = xgb_random.best_params_['n_estimators']

# Create more focused search space around best parameters
xgb_param_grid_focused = {
    'learning_rate': [max(0.01, best_lr - 0.02), best_lr, min(0.3, best_lr + 0.02)],
    'max_depth': [max(3, best_max_depth - 1), best_max_depth, min(12, best_max_depth + 1)],
    'min_child_weight': [max(1, best_min_child_weight - 1), best_min_child_weight, min(10, best_min_child_weight + 1)],
    'subsample': [max(0.5, best_subsample - 0.1), best_subsample, min(1.0, best_subsample + 0.1)],
    'colsample_bytree': [max(0.5, best_colsample_bytree - 0.1), best_colsample_bytree, min(1.0, best_colsample_bytree + 0.1)],
    'gamma': [max(0, best_gamma - 0.1), best_gamma, min(0.5, best_gamma + 0.1)],
    'n_estimators': [max(50, best_n_estimators - 50), best_n_estimators, min(500, best_n_estimators + 50)]
}

xgb_grid_focused = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_grid=xgb_param_grid_focused,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=1
)

xgb_grid_focused.fit(X_train, y_train)
print(f"Stage 2 Best XGB parameters: {xgb_grid_focused.best_params_}")
print(f"Stage 2 Best XGB score: {xgb_grid_focused.best_score_}")

# Step 4: Create and evaluate final models with best parameters
print("\nEvaluating tuned models...")

# Random Forest model with best parameters
rf_tuned = RandomForestClassifier(
    random_state=42,
    **rf_grid_focused.best_params_
)

# XGBoost model with best parameters
xgb_tuned = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    **xgb_grid_focused.best_params_
)

# Evaluate the models
results.append(evaluate_model(rf_tuned, "RF Advanced Tuned", X_train, X_test, y_train, y_test))
results.append(evaluate_model(xgb_tuned, "XGB Advanced Tuned", X_train, X_test, y_train, y_test))

# Step 5: Ensemble Method - Stacking
print("\nImplementing model stacking...")

# Use predictions from both models as features for a meta-classifier
rf_train_preds = rf_tuned.predict_proba(X_train)[:, 1]
xgb_train_preds = xgb_tuned.predict_proba(X_train)[:, 1]

# Create meta-features
meta_features_train = np.column_stack([rf_train_preds, xgb_train_preds])

# Meta-classifier
meta_classifier = XGBClassifier(
    learning_rate=0.05,
    max_depth=3,
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Train meta-classifier
meta_classifier.fit(meta_features_train, y_train)

# Predict on test set
rf_test_preds = rf_tuned.predict_proba(X_test)[:, 1]
xgb_test_preds = xgb_tuned.predict_proba(X_test)[:, 1]
meta_features_test = np.column_stack([rf_test_preds, xgb_test_preds])
meta_preds = meta_classifier.predict(meta_features_test)

# Evaluate stacked model
acc = accuracy_score(y_test, meta_preds)
prec = precision_score(y_test, meta_preds, average="weighted", zero_division=0)
rec = recall_score(y_test, meta_preds, average="weighted", zero_division=0)
f1 = f1_score(y_test, meta_preds, average="weighted", zero_division=0)

results.append({
    'Model': "Stacked Ensemble",
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1 Score': f1,
    'Training Time (s)': None,  # Not applicable for ensemble
    'Inference Time (s)': None   # Not applicable for ensemble
})

# Step 6: Voting Classifier (Simple averaging)
print("Implementing voting ensemble...")
voting_preds = (rf_test_preds + xgb_test_preds) / 2
voting_preds_binary = (voting_preds > 0.5).astype(int)

acc = accuracy_score(y_test, voting_preds_binary)
prec = precision_score(y_test, voting_preds_binary, average="weighted", zero_division=0)
rec = recall_score(y_test, voting_preds_binary, average="weighted", zero_division=0)
f1 = f1_score(y_test, voting_preds_binary, average="weighted", zero_division=0)

results.append({
    'Model': "Voting Ensemble",
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1 Score': f1,
    'Training Time (s)': None,  # Not applicable for ensemble
    'Inference Time (s)': None   # Not applicable for ensemble
})

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("advanced_tuning_results/advanced_tuning_results.csv", index=False)

# Create visualization
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='Model', y='F1 Score')
plt.title('F1 Score Comparison: Default vs. Advanced Tuning Methods')
plt.ylim(0.85, 0.95)  # Adjust as needed
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("advanced_tuning_plots/f1_score_comparison.png")

# Accuracy comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='Model', y='Accuracy')
plt.title('Accuracy Comparison: Default vs. Advanced Tuning Methods')
plt.ylim(0.85, 0.95)  # Adjust as needed
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("advanced_tuning_plots/accuracy_comparison.png")

# Save best parameters
best_params = {
    'RF_Stage1': rf_random.best_params_,
    'RF_Stage2': rf_grid_focused.best_params_,
    'XGB_Stage1': xgb_random.best_params_,
    'XGB_Stage2': xgb_grid_focused.best_params_
}

# Convert to DataFrame and save
pd.DataFrame([best_params]).to_csv("advanced_tuning_results/advanced_best_parameters.csv", index=False)

print("\nAdvanced hyperparameter tuning experiment completed!")
print("Results saved to 'advanced_tuning_results/advanced_tuning_results.csv'")
print("Plots saved to 'advanced_tuning_plots/' directory")

# Final recommendations
print("\n====== RECOMMENDATIONS ======")
best_model_idx = results_df['F1 Score'].idxmax()
best_model = results_df.iloc[best_model_idx]
print(f"Best model: {best_model['Model']} with F1 Score: {best_model['F1 Score']:.4f}")

if best_model['Model'] == "RF Advanced Tuned":
    print(f"Recommended RF parameters: {rf_grid_focused.best_params_}")
elif best_model['Model'] == "XGB Advanced Tuned":
    print(f"Recommended XGB parameters: {xgb_grid_focused.best_params_}")
elif best_model['Model'] in ["Stacked Ensemble", "Voting Ensemble"]:
    print("Recommended approach: Use ensemble method combining RF and XGB")
    print(f"RF parameters: {rf_grid_focused.best_params_}")
    print(f"XGB parameters: {xgb_grid_focused.best_params_}")
else:
    print(f"Recommended approach: Use default parameters as they perform best")

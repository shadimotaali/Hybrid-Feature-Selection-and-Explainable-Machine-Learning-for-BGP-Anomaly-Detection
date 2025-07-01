import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import os
import time
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

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
# Stage 1: Broad Bayesian search
print("\nPerforming Stage 1 (Broad) Bayesian optimization for Random Forest...")
rf_bayes_space = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(10, 100),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'max_features': Categorical(['sqrt', 'log2', None])
}

rf_bayes = BayesSearchCV(
    RandomForestClassifier(random_state=42),
    rf_bayes_space,
    n_iter=25,  # Limit iterations for first stage
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

rf_bayes.fit(X_train, y_train)
print(f"Stage 1 Best RF parameters: {rf_bayes.best_params_}")
print(f"Stage 1 Best RF score: {rf_bayes.best_score_}")

# Stage 2: Focused Bayesian search around the best parameters from Stage 1
print("\nPerforming Stage 2 (Focused) Bayesian optimization for Random Forest...")
best_n_estimators = rf_bayes.best_params_['n_estimators']
best_max_depth = rf_bayes.best_params_['max_depth']
best_min_samples_split = rf_bayes.best_params_['min_samples_split']
best_min_samples_leaf = rf_bayes.best_params_['min_samples_leaf']
best_max_features = rf_bayes.best_params_['max_features']

# Create more focused search space around best parameters
rf_bayes_space_focused = {
    'n_estimators': Integer(max(50, best_n_estimators - 50), best_n_estimators + 50),
    'max_depth': Integer(max(10, best_max_depth - 10), best_max_depth + 10),
    'min_samples_split': Integer(max(2, best_min_samples_split - 2), best_min_samples_split + 2),
    'min_samples_leaf': Integer(max(1, best_min_samples_leaf - 2), best_min_samples_leaf + 2),
    'max_features': Categorical([best_max_features])
}

rf_bayes_focused = BayesSearchCV(
    RandomForestClassifier(random_state=42),
    rf_bayes_space_focused,
    n_iter=25,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

rf_bayes_focused.fit(X_train, y_train)
print(f"Stage 2 Best RF parameters: {rf_bayes_focused.best_params_}")
print(f"Stage 2 Best RF score: {rf_bayes_focused.best_score_}")

# Step 3: Two-Stage Tuning for XGBoost
# Stage 1: Broad Bayesian search
print("\nPerforming Stage 1 (Broad) Bayesian optimization for XGBoost...")
xgb_bayes_space = {
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 12),
    'min_child_weight': Integer(1, 10),
    'subsample': Real(0.5, 1.0, prior='uniform'),
    'colsample_bytree': Real(0.5, 1.0, prior='uniform'),
    'gamma': Real(0, 0.5, prior='uniform'),
    'n_estimators': Integer(50, 500)
}

xgb_bayes = BayesSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    xgb_bayes_space,
    n_iter=25,  # Limit iterations for first stage
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

xgb_bayes.fit(X_train, y_train)
print(f"Stage 1 Best XGB parameters: {xgb_bayes.best_params_}")
print(f"Stage 1 Best XGB score: {xgb_bayes.best_score_}")

# Stage 2: Focused Bayesian search around the best parameters from Stage 1
print("\nPerforming Stage 2 (Focused) Bayesian optimization for XGBoost...")
best_lr = xgb_bayes.best_params_['learning_rate']
best_max_depth = xgb_bayes.best_params_['max_depth']
best_min_child_weight = xgb_bayes.best_params_['min_child_weight']
best_subsample = xgb_bayes.best_params_['subsample']
best_colsample_bytree = xgb_bayes.best_params_['colsample_bytree']
best_gamma = xgb_bayes.best_params_['gamma']
best_n_estimators = xgb_bayes.best_params_['n_estimators']

# Create more focused search space around best parameters
xgb_bayes_space_focused = {
    'learning_rate': Real(max(0.01, best_lr * 0.8), best_lr * 1.2, prior='log-uniform'),
    'max_depth': Integer(max(3, best_max_depth - 1), best_max_depth + 1),
    'min_child_weight': Integer(max(1, best_min_child_weight - 1), best_min_child_weight + 1),
    'subsample': Real(max(0.5, best_subsample - 0.1), min(1.0, best_subsample + 0.1), prior='uniform'),
    'colsample_bytree': Real(max(0.5, best_colsample_bytree - 0.1), min(1.0, best_colsample_bytree + 0.1), prior='uniform'),
    'gamma': Real(max(0, best_gamma - 0.05), best_gamma + 0.05, prior='uniform'),
    'n_estimators': Integer(max(50, best_n_estimators - 50), best_n_estimators + 50)
}

xgb_bayes_focused = BayesSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    xgb_bayes_space_focused,
    n_iter=25,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

xgb_bayes_focused.fit(X_train, y_train)
print(f"Stage 2 Best XGB parameters: {xgb_bayes_focused.best_params_}")
print(f"Stage 2 Best XGB score: {xgb_bayes_focused.best_score_}")

# Step 4: Create and evaluate final models with best parameters
print("\nEvaluating tuned models...")

# Random Forest model with best parameters
rf_tuned = RandomForestClassifier(
    random_state=42,
    **rf_bayes_focused.best_params_
)

# XGBoost model with best parameters
xgb_tuned = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    **xgb_bayes_focused.best_params_
)

# Evaluate the models
results.append(evaluate_model(rf_tuned, "RF Bayesian Tuned", X_train, X_test, y_train, y_test))
results.append(evaluate_model(xgb_tuned, "XGB Bayesian Tuned", X_train, X_test, y_train, y_test))

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
    'RF_Bayesian_Stage1': rf_bayes.best_params_,
    'RF_Bayesian_Stage2': rf_bayes_focused.best_params_,
    'XGB_Bayesian_Stage1': xgb_bayes.best_params_,
    'XGB_Bayesian_Stage2': xgb_bayes_focused.best_params_
}

# Convert to DataFrame and save
pd.DataFrame([best_params]).to_csv("advanced_tuning_results/bayesian_best_parameters.csv", index=False)

print("\nAdvanced hyperparameter tuning experiment completed!")
print("Results saved to 'advanced_tuning_results/advanced_tuning_results.csv'")
print("Plots saved to 'advanced_tuning_plots/' directory")

# Final recommendations
print("\n====== RECOMMENDATIONS ======")
best_model_idx = results_df['F1 Score'].idxmax()
best_model = results_df.iloc[best_model_idx]
print(f"Best model: {best_model['Model']} with F1 Score: {best_model['F1 Score']:.4f}")

if best_model['Model'] == "RF Bayesian Tuned":
    print(f"Recommended RF parameters: {rf_bayes_focused.best_params_}")
elif best_model['Model'] == "XGB Bayesian Tuned":
    print(f"Recommended XGB parameters: {xgb_bayes_focused.best_params_}")
elif best_model['Model'] in ["Stacked Ensemble", "Voting Ensemble"]:
    print("Recommended approach: Use ensemble method combining RF and XGB")
    print(f"RF parameters: {rf_bayes_focused.best_params_}")
    print(f"XGB parameters: {xgb_bayes_focused.best_params_}")
else:
    print(f"Recommended approach: Use default parameters as they perform best")

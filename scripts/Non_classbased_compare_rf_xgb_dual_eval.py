import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from matplotlib.ticker import MaxNLocator

# -----------------------------
# 1. Setup paths and models
# -----------------------------

print("\n📥 Loading full dataset...")
data = pd.read_excel("Randomly_Balanced_Dataset.xlsx")

# Drop datetime columns
datetime_cols = data.select_dtypes(include=["datetime64[ns]", "object"]).columns
data = data.drop(columns=[col for col in datetime_cols if "timestamp" in col.lower()])

X_full = data.drop(columns=["target"])
y = data["target"]

# Create output folders
os.makedirs("plots", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# Define feature selection files to use
feature_selection_methods = {


    "Random Forest Importance": "feature_selection_results/Random_Forest_Importance.csv",
    "XGBoost Importance": "feature_selection_results/XGBoost_Importance.csv"
}

# Define models
model_map = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# Cross-validation config
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall': make_scorer(recall_score, average='weighted', zero_division=0),
    'f1': make_scorer(f1_score, average='weighted', zero_division=0)
}

# -----------------------------
# 2. Run Evaluation
# -----------------------------

all_results = []

for method_name, feature_file in feature_selection_methods.items():
    print(f"\n🔎 Processing feature selection method: {method_name}")

    top_features = pd.read_csv(feature_file)["Feature"].tolist()

    for model_name, model in model_map.items():

        for k in range(48, 0, -5):
            selected_features = top_features[:k]
            X_selected = X_full[selected_features]

            # Train-test split evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, stratify=y, random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc_tt = accuracy_score(y_test, y_pred)
            prec_tt = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec_tt = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1_tt = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            # Cross-validation evaluation
            scores = cross_validate(model, X_selected, y, cv=cv, scoring=scoring)

            acc_cv = scores['test_accuracy'].mean()
            prec_cv = scores['test_precision'].mean()
            rec_cv = scores['test_recall'].mean()
            f1_cv = scores['test_f1'].mean()

            # Save results
            all_results.append({
                "Feature_Selection_Method": method_name,
                "Model": model_name,
                "Features": k,
                "Eval": "Train-Test",
                "Accuracy": acc_tt,
                "Precision": prec_tt,
                "Recall": rec_tt,
                "F1 Score": f1_tt
            })

            all_results.append({
                "Feature_Selection_Method": method_name,
                "Model": model_name,
                "Features": k,
                "Eval": "Cross-Validation",
                "Accuracy": acc_cv,
                "Precision": prec_cv,
                "Recall": rec_cv,
                "F1 Score": f1_cv
            })

# -----------------------------
# 3. Save Results
# -----------------------------

print("\n💾 Saving combined evaluation results...")
df_results = pd.DataFrame(all_results)
df_results.to_csv("metrics/compare_rf_xgb_per_feature_selection.csv", index=False)

# -----------------------------
# 4. Plot Results
# -----------------------------

print("\n📊 Saving plots...")
metrics_list = ["Accuracy", "Precision", "Recall", "F1 Score"]

for method in df_results["Feature_Selection_Method"].unique():
    df_sub = df_results[df_results["Feature_Selection_Method"] == method]

    for metric in metrics_list:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df_sub,
            x="Features",
            y=metric,
            hue="Eval",
            style="Model",
            markers=True,
            dashes=False
        )
        plt.title(f"{metric} vs Features ({method})")
        plt.xlabel("Number of Features")
        plt.ylabel(metric)
        plt.grid(True)
        plt.tight_layout()
        safe_method = method.replace(" ", "_").replace("-", "_")
        plt.savefig(f"plots/{safe_method.lower()}_{metric.lower().replace(' ', '_')}.png")
        plt.close()

print("\n✅ All results saved inside 'metrics/' and 'plots/' directories.")

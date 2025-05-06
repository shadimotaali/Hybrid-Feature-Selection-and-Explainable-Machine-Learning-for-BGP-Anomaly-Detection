
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Dataset and model configuration
datasets = {
    "Class_1": "Class_1.xlsx",
    "Class_2": "Class_2.xlsx",
    "Class_3": "Class_3.xlsx"
}

results = []

# Iterate over datasets
for dataset_name, dataset_path in datasets.items():
    data = pd.read_excel(dataset_path)

    # Drop datetime/timestamp columns
    datetime_cols = data.select_dtypes(include=["datetime64[ns]", "datetime64", "object"]).columns
    data = data.drop(columns=[col for col in datetime_cols if "timestamp" in col.lower()])

    # Normalize labels to binary
    data["target"] = data["target"].apply(lambda x: 0 if x == 0 else 1)

    X = data.drop(columns=["target"])
    y = data["target"]

    model_map = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # Evaluate each model
    for model_name, model in model_map.items():
        feature_path = f"feature_selection_{dataset_name}/{model_name.replace(' ', '_')}_top_48.csv"
        top_features = pd.read_csv(feature_path)["Feature"].tolist()

        for k in range(48, 0, -5):
            selected_features = top_features[:k]
            X_selected = X[selected_features]

            # Train-Test split evaluation
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc_tt = accuracy_score(y_test, y_pred)
            prec_tt = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec_tt = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1_tt = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            # Cross-validation evaluation
            acc_cv = cross_val_score(model, X_selected, y, cv=5, scoring="accuracy").mean()
            f1_cv = cross_val_score(model, X_selected, y, cv=5, scoring="f1_weighted").mean()
            prec_cv = cross_val_score(model, X_selected, y, cv=5, scoring="precision_weighted").mean()
            rec_cv = cross_val_score(model, X_selected, y, cv=5, scoring="recall_weighted").mean()

            results.extend([
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Features": k,
                    "Eval": "Train-Test",
                    "Accuracy": acc_tt,
                    "Precision": prec_tt,
                    "Recall": rec_tt,
                    "F1 Score": f1_tt
                },
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Features": k,
                    "Eval": "Cross-Validation",
                    "Accuracy": acc_cv,
                    "Precision": prec_cv,
                    "Recall": rec_cv,
                    "F1 Score": f1_cv
                }
            ])

# Save results to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("separate_class_rf_xgb_dual_eval.csv", index=False)

# Plotting section
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
os.makedirs("plots", exist_ok=True)

for metric in metrics:
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_results, x="Features", y=metric, hue="Eval",
        style="Model", markers=True, dashes=False, palette="Set2", hue_order=["Train-Test", "Cross-Validation"]
    )
    plt.title(f"{metric} vs Number of Features (All Classes)")
    plt.xlabel("Number of Features")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend(title="Evaluation")
    plt.tight_layout()
    plt.savefig(f"plots/{metric.lower().replace(' ', '_')}_combined_all_classes.png")
    plt.close()

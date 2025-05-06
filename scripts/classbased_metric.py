import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Input datasets
datasets = {
    "Class_1": "Class_1.xlsx",
    "Class_2": "Class_2.xlsx",
    "Class_3": "Class_3.xlsx"
}

# Storage for per-class results
results = []

for dataset_name, path in datasets.items():
    print(f"Processing {dataset_name}...")

    data = pd.read_excel(path)
    datetime_cols = data.select_dtypes(include=["datetime64[ns]", "datetime64", "object"]).columns
    data = data.drop(columns=[col for col in datetime_cols if "timestamp" in col.lower()])
    data["target"] = data["target"].apply(lambda x: 0 if x == 0 else 1)
    X = data.drop(columns=["target"])
    y = data["target"]

    # Feature directory per class
    features_dir = f"./feature_selection_{dataset_name}"
    rf_features = pd.read_csv(f"{features_dir}/Random_Forest_top_48.csv")["Feature"].tolist()
    xgb_features = pd.read_csv(f"{features_dir}/XGBoost_top_48.csv")["Feature"].tolist()

    models = {
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42),
            "features": rf_features
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "features": xgb_features
        }
    }

    for model_name, model_info in models.items():
        model = model_info["model"]
        features = model_info["features"][:48]  # Ensure top 48
        X_selected = X[features]
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, stratify=y, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        })

# Save results
df = pd.DataFrame(results)
df.to_csv("per_class_metrics_summary.csv", index=False)

# Create plots
os.makedirs("per_class_plots", exist_ok=True)

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Dataset", y=metric, hue="Model")
    plt.title(f"{metric} Per Class")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"per_class_plots/{metric.lower().replace(' ', '_')}_per_class.png")
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Dataset names
datasets = {
    "Class_1": ("Class_1.xlsx", "Class_1 - Hybrid.xlsx"),
    "Class_2": ("Class_2.xlsx", "Class_2 - Hybrid.xlsx"),
    "Class_3": ("Class_3.xlsx", "Class_3 - Hybrid.xlsx"),
    "Full_Dataset": ("Randomly_Balanced_Dataset.xlsx", "Randomly_Balanced_Dataset - Hybrid.xlsx")
}

# Feature paths
full_feature_paths = {
    "Class_1": ("./feature_selection_Class_1/Random_Forest_top_48.csv", "./feature_selection_Class_1/XGBoost_top_48.csv"),
    "Class_2": ("./feature_selection_Class_2/Random_Forest_top_48.csv", "./feature_selection_Class_2/XGBoost_top_48.csv"),
    "Class_3": ("./feature_selection_Class_3/Random_Forest_top_48.csv", "./feature_selection_Class_3/XGBoost_top_48.csv"),
    "Full_Dataset": ("./feature_selection_results/Random_Forest_Importance.csv", "./feature_selection_results/XGBoost_Importance.csv")
}

# Hybrid feature list
hybrid_features = pd.read_csv("feature_selection_results/Hybrid_top_features.csv")["Feature"].tolist()

# Create output directory
os.makedirs("confusion_matrix_comparisons", exist_ok=True)

for dataset_name, (full_path, hybrid_path) in datasets.items():
    print(f"📊 Comparing confusion matrices for {dataset_name}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{dataset_name} – Confusion Matrices (Full vs Hybrid)", fontsize=16)

    # Load and prepare both datasets
    for i, (feature_type, file_path) in enumerate([("Full", full_path), ("Hybrid", hybrid_path)]):
        df = pd.read_excel(file_path)
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64", "object"]).columns
        df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

        X = df.drop(columns=["target"])
        y = df["target"]

        if feature_type == "Full":
            rf_features = pd.read_csv(full_feature_paths[dataset_name][0])["Feature"].tolist()
            xgb_features = pd.read_csv(full_feature_paths[dataset_name][1])["Feature"].tolist()
        else:
            rf_features = xgb_features = hybrid_features

        models = {
            "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), rf_features),
            "XGBoost": (XGBClassifier(eval_metric='logloss', random_state=42), xgb_features)
        }

        for j, (model_name, (model, selected_features)) in enumerate(models.items()):
            X_selected = X[selected_features]
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
            row = i
            col = j
            disp.plot(ax=axes[row][col], cmap='Blues', colorbar=False)
            axes[row][col].set_title(f"{model_name} ({feature_type})")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"confusion_matrix_comparisons/{dataset_name}_comparison.png")
    plt.close()

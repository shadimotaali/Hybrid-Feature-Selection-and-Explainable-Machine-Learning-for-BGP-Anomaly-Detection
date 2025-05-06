import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

# --- Setup output folders ---
os.makedirs("model_results", exist_ok=True)
os.makedirs("pca_tsne_plots", exist_ok=True)

# --- Dataset paths ---
dataset_files = {
    "Class_1": "Class_1 - Hybrid.xlsx",
    "Class_2": "Class_2 - Hybrid.xlsx",
    "Class_3": "Class_3 - Hybrid.xlsx",
    "Full_Dataset": "Randomly_Balanced_Dataset - Hybrid.xlsx"
}

# --- Models ---
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# --- Results storage ---
results = []

# --- Main loop ---
for dataset_name, dataset_path in dataset_files.items():
    print(f"\n📥 Processing {dataset_name}...")

    df = pd.read_excel(dataset_path)

    # Drop any datetime columns
    datetime_cols = df.select_dtypes(include=["datetime64", "object"]).columns
    df = df.drop(columns=[col for col in datetime_cols if "timestamp" in col.lower()], errors="ignore")

    # Separate features and label
    X = df.drop(columns=["target"])
    y = df["target"]
    y = y.apply(lambda label: 0 if label == 0 else 1)
    for model_name, model in models.items():
        print(f"\n🔵 Training {model_name} on {dataset_name}...")

        # Train-Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })

        # --- PCA plot ---
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="Set1")
        plt.title(f"PCA - {model_name} ({dataset_name})")
        plt.tight_layout()
        plt.savefig(f"pca_tsne_plots/PCA_{model_name.replace(' ', '_')}_{dataset_name}.png")
        plt.close()

        # --- t-SNE plot ---
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="Set1")
        plt.title(f"t-SNE - {model_name} ({dataset_name})")
        plt.tight_layout()
        plt.savefig(f"pca_tsne_plots/TSNE_{model_name.replace(' ', '_')}_{dataset_name}.png")
        plt.close()

# --- Save overall results ---
results_df = pd.DataFrame(results)
results_df.to_csv("model_results/hybrid_models_evaluation.csv", index=False)

print("\n✅ All model evaluations and plots completed!")

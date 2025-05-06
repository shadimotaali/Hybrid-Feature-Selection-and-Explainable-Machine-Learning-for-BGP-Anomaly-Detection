import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Dataset file paths
dataset_files = {
    "Class_1": "Class_1.xlsx",
    "Class_2": "Class_2.xlsx",
    "Class_3": "Class_3.xlsx"
}

# Output directory
output_dir = "shap_summary_plots_xgb"
os.makedirs(output_dir, exist_ok=True)

# Loop through each class dataset
for class_name, file_path in dataset_files.items():
    print(f"🔍 Processing {class_name}...")

    # Load dataset
    df = pd.read_excel(file_path)

    # Drop datetime columns if any
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "object"]).columns
    df = df.drop(columns=[col for col in datetime_cols if "timestamp" in col.lower()])

    # Load class-specific top-48 features for XGBoost
    feature_path = f"feature_selection_{class_name}/XGBoost_top_48.csv"
    top_features = pd.read_csv(feature_path)["Feature"].tolist()

    # Prepare X and y
    main_class = df["target"].mode()[0]
    y = (df["target"] == main_class).astype(int)
    X = df[top_features]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train XGBoost model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    # SHAP with TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)   # ⚡ NO [1] here

    # Bar summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_bar_{class_name}.png")
    plt.close()

    # Dot summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_dot_{class_name}.png")
    plt.close()

print(f"\n✅ SHAP summary plots (XGBoost) saved to: {output_dir}/")

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create output folder
os.makedirs("gini_dotplots", exist_ok=True)

# Define datasets
datasets = {
    "Class_1": "Class_1 - Hybrid.xlsx",
    "Class_2": "Class_2 - Hybrid.xlsx",
    "Class_3": "Class_3 - Hybrid.xlsx"
}

print("\n🌲 Generating Gini Index dot plots for Random Forest models...\n")

for label, file_path in datasets.items():
    print(f"🔍 Processing {label}...")

    df = pd.read_excel(file_path)

    # Drop non-numeric and timestamp columns
    df = df.select_dtypes(include=["number"])
    X = df.drop(columns=["target"])
    y = df["target"]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Get Gini importances
    importances = model.feature_importances_
    feature_names = X.columns

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Gini Importance": importances
    }).sort_values(by="Gini Importance", ascending=False)

    # Plot dot plot
    plt.figure(figsize=(8, 10))
    sns.stripplot(x="Gini Importance", y="Feature", data=df_imp, color="darkgreen", orient="h", size=8)
    plt.title(f"Gini Index Dot Plot - {label} (Random Forest)", fontsize=14)
    plt.tight_layout()
    plot_path = f"gini_dotplots/gini_dotplot_{label}.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"✅ Saved: {plot_path}")

print("\n🎯 All Gini dot plots saved in 'gini_dotplots/' directory.\n")

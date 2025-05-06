import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# --- Load Hybrid Dataset ---
data = pd.read_excel("Randomly_Balanced_Dataset - Hybrid.xlsx")
X = data.drop(columns=["target"])
y = data["target"]

# --- Load Hybrid Features ---
hybrid_features = pd.read_csv("feature_selection_results/Hybrid_top_features.csv")["Feature"].tolist()

# --- Define Models ---
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# --- Define Top-N Subsets ---
top_n_list = [10, 20, 25]

# --- Evaluate Each Model and Top-N Features ---
results = []

for n in top_n_list:
    selected_features = hybrid_features[:n]
    X_selected = X[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        results.append({
            "Feature Subset": f"Top-{n}",
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "F1 Score": round(f1, 4)
        })

# --- Output Table ---
df = pd.DataFrame(results)
print(df)

# --- Optional: Save results ---
df.to_csv("ablation_study_results.csv", index=False)

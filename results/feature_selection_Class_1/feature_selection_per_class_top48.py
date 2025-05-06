
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest, RFE
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Files and class names
datasets = {
    "Class_1": "Class_1.xlsx",
    "Class_2": "Class_2.xlsx",
    "Class_3": "Class_3.xlsx"
}

methods = [
    ("ANOVA_F_score", lambda X, y: SelectKBest(score_func=f_classif, k='all').fit(X, y).scores_),
    ("Mutual_Information", lambda X, y: SelectKBest(score_func=mutual_info_classif, k='all').fit(X, y).scores_),
    ("Random_Forest", lambda X, y: RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y).feature_importances_),
    ("XGBoost", lambda X, y: XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42).fit(X, y).feature_importances_),
    ("Lasso_Regression", lambda X, y: np.abs(LassoCV(cv=5, random_state=42).fit(X, y).coef_)),
    ("RFE", lambda X, y: RFE(LogisticRegression(max_iter=1000, solver='liblinear'), n_features_to_select=min(48, X.shape[1])).fit(X, y).ranking_ * -1),
    ("Gini_Decision_Tree", lambda X, y: DecisionTreeClassifier(criterion="gini", random_state=42).fit(X, y).feature_importances_),
]

for name, file in datasets.items():
    df = pd.read_excel(file)
    df = df.drop(columns=["timestamp", "timestamp2", "source_file"])
    df["binary_target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)
    X = df.drop(columns=["target", "binary_target"])
    y = df["binary_target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    os.makedirs(f"feature_selection_{name}", exist_ok=True)

    for method_name, method_func in methods:
        try:
            scores = method_func(X_train, y_train)
            n_top = min(48, len(scores))
            indices = np.argsort(scores)[::-1][:n_top]
            top_features = X.columns[indices]
            top_scores = scores[indices]

            # Save CSV
            pd.DataFrame({"Feature": top_features, "Score": top_scores}).to_csv(
                f"feature_selection_{name}/{method_name}_top_48.csv", index=False
            )

            # Plot
            plt.figure(figsize=(10, 12))
            plt.title(f"{name}: Top 48 Features by {method_name}")
            plt.barh(range(n_top), top_scores[::-1], align='center')
            plt.yticks(range(n_top), top_features[::-1])
            plt.xlabel("Importance Score")
            plt.tight_layout()
            plt.savefig(f"feature_selection_{name}/{method_name}_importance.png")
            plt.close()
        except Exception as e:
            print(f"Error in {method_name} for {name}: {e}")

print("✅ Per-class feature selection completed for Class_1, Class_2, Class_3")

import shap
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load dataset and top-48 features
data = pd.read_excel("Randomly_Balanced_Dataset.xlsx")
data = data.drop(columns=[col for col in data.select_dtypes(include=["datetime64[ns]", "object"]).columns if "timestamp" in col.lower()])
X = data.drop(columns=["target"])
y = data["target"]

# Load top-48 features from XGBoost
top_features_df = pd.read_csv("feature_selection_results/XGBoost_top_48.csv")

top_features = top_features_df["Feature"].tolist()

# Filter dataset by top features
X_top = X[top_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, stratify=y, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)


# Plot summary
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("xgboost_shap_summary_plot.png")
plt.close()

# Return the path of the saved image
"xgboost_shap_summary_plot.png"

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.feature_selection import f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

# Create output directory if not exists
output_dir = "feature_selection_results"
os.makedirs(output_dir, exist_ok=True)

# ---------------------
# STEP 1: Load the data
# ---------------------
print("\n📥 Loading dataset...")
df = pd.read_excel("Randomly_Balanced_Dataset.xlsx")

# ---------------------
# STEP 2: Clean the data
# ---------------------
print("\n🧹 Cleaning data...")
X = X.select_dtypes(include=['number'])
X = X.fillna(0)
y = df['target']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Timing dictionary
timings = {}

# -----------------------------
# STEP 3: ANOVA F-score method
# -----------------------------
start = time.time()
f_scores, _ = f_classif(X_scaled, y)
f_score_df = pd.DataFrame({'Feature': X.columns, 'ANOVA_F_score': f_scores}).sort_values(by='ANOVA_F_score', ascending=False)

# Save CSV and plot
f_score_df.to_csv(f"{output_dir}/ANOVA_F_score.csv", index=False)
plt.figure(figsize=(12,6))
sns.barplot(x='ANOVA_F_score', y='Feature', data=f_score_df.head(20))
plt.title("Top 20 Features by ANOVA F-score")
plt.tight_layout()
plt.savefig(f"{output_dir}/ANOVA_F_score_top20.png")
plt.close()

timings['ANOVA F-score'] = time.time() - start

# ------------------------------------------
# STEP 4: Mutual Information Feature Scores
# ------------------------------------------
start = time.time()
mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual_Information': mi_scores}).sort_values(by='Mutual_Information', ascending=False)

mi_df.to_csv(f"{output_dir}/Mutual_Information.csv", index=False)
plt.figure(figsize=(12,6))
sns.barplot(x='Mutual_Information', y='Feature', data=mi_df.head(20))
plt.title("Top 20 Features by Mutual Information")
plt.tight_layout()
plt.savefig(f"{output_dir}/Mutual_Information_top20.png")
plt.close()

timings['Mutual Information'] = time.time() - start

# -----------------------------
# STEP 5: Random Forest method
# -----------------------------
start = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_scaled, y)
rf_importances = rf.feature_importances_
rf_df = pd.DataFrame({'Feature': X.columns, 'Random_Forest_Importance': rf_importances}).sort_values(by='Random_Forest_Importance', ascending=False)

rf_df.to_csv(f"{output_dir}/Random_Forest_Importance.csv", index=False)
plt.figure(figsize=(12,6))
sns.barplot(x='Random_Forest_Importance', y='Feature', data=rf_df.head(20))
plt.title("Top 20 Features by Random Forest Importance")
plt.tight_layout()
plt.savefig(f"{output_dir}/Random_Forest_Importance_top20.png")
plt.close()

timings['Random Forest'] = time.time() - start

# -----------------------------
# STEP 6: XGBoost Feature Importances
# -----------------------------
start = time.time()
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_scaled, y)
xgb_importances = xgb.feature_importances_
xgb_df = pd.DataFrame({'Feature': X.columns, 'XGBoost_Importance': xgb_importances}).sort_values(by='XGBoost_Importance', ascending=False)

xgb_df.to_csv(f"{output_dir}/XGBoost_Importance.csv", index=False)
plt.figure(figsize=(12,6))
sns.barplot(x='XGBoost_Importance', y='Feature', data=xgb_df.head(20))
plt.title("Top 20 Features by XGBoost Importance")
plt.tight_layout()
plt.savefig(f"{output_dir}/XGBoost_Importance_top20.png")
plt.close()

timings['XGBoost'] = time.time() - start

# ------------------------------------------
# STEP 7: Recursive Feature Elimination (RFE)
# ------------------------------------------
start = time.time()
model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
rfe = RFE(model, n_features_to_select=10, step=2)
rfe.fit(X_scaled, y)
rfe_df = pd.DataFrame({
    'Feature': X.columns,
    'RFE_Ranking': rfe.ranking_
}).sort_values(by='RFE_Ranking')

rfe_df.to_csv(f"{output_dir}/RFE_Ranking.csv", index=False)
plt.figure(figsize=(12,6))
sns.barplot(x='RFE_Ranking', y='Feature', data=rfe_df.head(20))
plt.title("Top 20 Features by RFE")
plt.tight_layout()
plt.savefig(f"{output_dir}/RFE_Ranking_top20.png")
plt.close()

timings['RFE'] = time.time() - start

# ------------------------------------------
# STEP 8: Lasso Regression Feature Selection
# ------------------------------------------
start = time.time()
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)
lasso_df = pd.DataFrame({
    'Feature': X.columns,
    'Lasso_Coefficient': lasso.coef_
}).sort_values(by='Lasso_Coefficient', ascending=False)

lasso_df.to_csv(f"{output_dir}/Lasso_Coefficient.csv", index=False)
plt.figure(figsize=(12,6))
sns.barplot(x='Lasso_Coefficient', y='Feature', data=lasso_df.head(20))
plt.title("Top 20 Features by Lasso")
plt.tight_layout()
plt.savefig(f"{output_dir}/Lasso_Coefficient_top20.png")
plt.close()

timings['Lasso Regression'] = time.time() - start

# ------------------------------------------
# STEP 9: Completion Message
# ------------------------------------------
print("\n✅ All feature rankings completed!")
print("\n🕰️ Timing Summary:")
for method, secs in timings.items():
    print(f"{method}: {secs:.2f} seconds")

print(f"\n📂 All CSVs and PNG plots saved inside: {output_dir}/ ✅")

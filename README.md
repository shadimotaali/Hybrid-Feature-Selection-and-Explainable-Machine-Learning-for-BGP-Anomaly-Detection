# Hybrid-Feature-Selection-and-Explainable-Machine-Learning-for-BGP-Anomaly-Detection
Hybrid feature selection and explainable ML for BGP anomaly detection. Combines 6 selection methods and avoids synthetic oversampling to ensure data integrity. Uses SHAP, Gini, PCA, and t-SNE for interpretation. Tested on real BGP data with RF and XGBoost.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Hybrid Feature Selection and Explainable Machine Learning for BGP Anomaly Detection

This repository contains the full implementation of the research paper titled **"Hybrid Feature Selection and Explainable Machine Learning for BGP Anomaly Detection"**. This work presents a robust and interpretable machine learning pipeline for detecting BGP anomalies using hybrid feature selection and explainable AI techniques.

## 📂 Repository Structure

- `datasets/`: Real-sample balanced datasets used for model training and evaluation.
- `scripts/`: Python scripts for feature engineering, and visualizations.
- `results/`: Output metrics, confusion matrices, and SHAP visualizations.
- `requirements.txt`: List of Python dependencies.

## 📌 Key Features

- Hybrid feature selection using 6 methods: ANOVA F-score, Mutual Information, Random Forest, XGBoost, RFE, and Lasso.
- Real-sample balanced dataset creation (avoiding SMOTE).
- Classification using Random Forest and XGBoost.
- Explainability via SHAP (XGBoost) and Gini Index (Random Forest).
- Dimensionality reduction using t-SNE and PCA.

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shadimotaali/Hybrid-Feature-Selection-and-Explainable-Machine-Learning-for-BGP-Anomaly-Detection.git
   cd Hybrid-Feature-Selection-and-Explainable-Machine-Learning-for-BGP-Anomaly-Detection

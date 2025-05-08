import pandas as pd
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Dataset paths
datasets = {
    "All_Classes": "Randomly_Balanced_Dataset.xlsx",
    "Class_1": "Class_1.xlsx",
    "Class_2": "Class_2.xlsx",
    "Class_3": "Class_3.xlsx"
}

output_dir = "dimensionality_reduction_plots"
os.makedirs(output_dir, exist_ok=True)

for name, file in datasets.items():
    df = pd.read_excel(file)
    features = df.drop(columns=["target"], errors="ignore")
    labels = df["target"] if "target" in df.columns else None

    # Standardize features
    X_scaled = StandardScaler().fit_transform(features)

    # >>> PCA variance analysis
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumulative_variance = pca_full.explained_variance_ratio_.cumsum()
    n_components_99 = (cumulative_variance < 0.99).sum() + 1
    print(f"{name}: {n_components_99} components cover 99% of the variance.")
    # <<< End PCA variance analysis

    # Consistent color map
    unique_labels = sorted(labels.unique())
    palette = sns.color_palette("Set2", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    label_colors = [color_map[label] for label in labels]

    # PCA 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_ * 100
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette=color_map, s=60)
    plt.xlabel(f"PC1 ({var_ratio[0]:.1f}% Variance)")
    plt.ylabel(f"PC2 ({var_ratio[1]:.1f}% Variance)")
    plt.title(f"PCA (2D) - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_PCA_2D.png"))
    plt.close()

    # PCA 3D
    pca3 = PCA(n_components=3)
    X_pca3 = pca3.fit_transform(X_scaled)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for label in unique_labels:
        idx = labels == label
        ax.scatter(X_pca3[idx, 0], X_pca3[idx, 1], X_pca3[idx, 2], c=[color_map[label]], label=str(label), s=40)
    ax.set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca3.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca3.explained_variance_ratio_[2]*100:.1f}%)")
    plt.title(f"PCA (3D) - {name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_PCA_3D.png"))
    plt.close()




print("✅ All PCA plots (2D + 3D) and PCA explained variance printed.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import gc
import sys
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

print("BGP Anomaly Detection - Computational Efficiency Benchmark")
print("=" * 70)

# Function to measure memory usage (simple alternative to psutil)
def get_process_memory():
    try:
        # Try to use psutil if available (more accurate)
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # in MB
    except ImportError:
        # Fallback to sys.getsizeof for approximate measurement
        print("Note: psutil not installed. Using simplified memory tracking.")
        return 0  # Just return 0 and we'll skip memory measurements

# Hybrid feature list - for reference only (we'll use the actual hybrid dataset files)
try:
    hybrid_features = pd.read_csv("feature_selection_results/Hybrid_top_features.csv")["Feature"].tolist()[:25]
    print(f"Hybrid feature selection identified {len(hybrid_features)} features")
except FileNotFoundError:
    print("Hybrid features file not found. Continuing without reference list.")
    hybrid_features = []

# Dataset paths - both regular and hybrid versions
dataset_pairs = {
    "Class_1": ("Class_1.xlsx", "Class_1 - Hybrid.xlsx"),
    "Class_2": ("Class_2.xlsx", "Class_2 - Hybrid.xlsx"),
    "Class_3": ("Class_3.xlsx", "Class_3 - Hybrid.xlsx"),
    "Full_Dataset": ("Randomly_Balanced_Dataset.xlsx", "Randomly_Balanced_Dataset - Hybrid.xlsx")
}

# Models to evaluate
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Results storage
results = []

# Number of trials for averaging
num_trials = 5

for dataset_name, (regular_path, hybrid_path) in dataset_pairs.items():
    print(f"\nProcessing {dataset_name}...")
    
    try:
        # Load regular dataset
        df_regular = pd.read_excel(regular_path)
        
        # Select only numeric columns (excluding target)
        numeric_cols = df_regular.select_dtypes(include=['number']).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        # Ensure 'target' is in the dataframe
        if 'target' not in df_regular.columns:
            print(f"Error: 'target' column not found in {regular_path}. Skipping...")
            continue
            
        # Keep only numeric columns plus target
        df_regular = df_regular[numeric_cols + ['target']]
        
        # Drop timestamp columns if any
        df_regular = df_regular.drop(columns=[col for col in df_regular.columns if "timestamp" in str(col).lower()], errors="ignore")
        
        # Normalize labels to binary
        df_regular["target"] = df_regular["target"].apply(lambda x: 0 if x == 0 else 1)
        
        # Load hybrid dataset
        df_hybrid = pd.read_excel(hybrid_path)
        
        # Select only numeric columns (excluding target)
        numeric_cols = df_hybrid.select_dtypes(include=['number']).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
            
        # Keep only numeric columns plus target
        df_hybrid = df_hybrid[numeric_cols + ['target']]
        
        # Drop timestamp columns if any
        df_hybrid = df_hybrid.drop(columns=[col for col in df_hybrid.columns if "timestamp" in str(col).lower()], errors="ignore")
        
        # Normalize labels to binary
        df_hybrid["target"] = df_hybrid["target"].apply(lambda x: 0 if x == 0 else 1)
        
        # Prepare features and labels
        X_full = df_regular.drop(columns=["target"])
        X_hybrid = df_hybrid.drop(columns=["target"])
        y_regular = df_regular["target"]
        y_hybrid = df_hybrid["target"]
        
        print(f"  Regular dataset: {X_full.shape[1]} features")
        print(f"  Hybrid dataset: {X_hybrid.shape[1]} features")
        
        # Split data
        X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
            X_full, y_regular, test_size=0.2, stratify=y_regular, random_state=42
        )
        
        X_hybrid_train, X_hybrid_test, y_hybrid_train, y_hybrid_test = train_test_split(
            X_hybrid, y_hybrid, test_size=0.2, stratify=y_hybrid, random_state=42
        )
        
        for model_name, model_base in models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Full feature set metrics
            full_train_times = []
            full_inference_times = []
            full_memory_usages = []
            full_accuracies = []
            full_f1_scores = []
            
            # Hybrid feature set metrics
            hybrid_train_times = []
            hybrid_inference_times = []
            hybrid_memory_usages = []
            hybrid_accuracies = []
            hybrid_f1_scores = []
            
            for trial in range(num_trials):
                print(f"  Trial {trial+1}/{num_trials}...", end="", flush=True)
                
                # Reset models for each trial
                if model_name == "Random Forest":
                    full_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    hybrid_model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    full_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                    hybrid_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                
                # Full feature set
                # Force garbage collection
                gc.collect()
                
                # Measure training time and memory
                start_memory = get_process_memory()
                start_time = time.time()
                full_model.fit(X_full_train, y_full_train)
                end_time = time.time()
                end_memory = get_process_memory()
                
                full_train_times.append(end_time - start_time)
                full_memory_usages.append(end_memory - start_memory)
                
                # Measure inference time
                start_time = time.time()
                y_pred_full = full_model.predict(X_full_test)
                end_time = time.time()
                
                full_inference_times.append(end_time - start_time)
                full_accuracies.append(accuracy_score(y_full_test, y_pred_full))
                full_f1_scores.append(f1_score(y_full_test, y_pred_full))
                
                # Hybrid feature set
                # Force garbage collection
                gc.collect()
                
                # Measure training time and memory
                start_memory = get_process_memory()
                start_time = time.time()
                hybrid_model.fit(X_hybrid_train, y_hybrid_train)
                end_time = time.time()
                end_memory = get_process_memory()
                
                hybrid_train_times.append(end_time - start_time)
                hybrid_memory_usages.append(end_memory - start_memory)
                
                # Measure inference time
                start_time = time.time()
                y_pred_hybrid = hybrid_model.predict(X_hybrid_test)
                end_time = time.time()
                
                hybrid_inference_times.append(end_time - start_time)
                hybrid_accuracies.append(accuracy_score(y_hybrid_test, y_pred_hybrid))
                hybrid_f1_scores.append(f1_score(y_hybrid_test, y_pred_hybrid))
                
                print(" Done")
            
            # Calculate averages
            avg_full_train_time = np.mean(full_train_times)
            avg_full_inference_time = np.mean(full_inference_times)
            avg_full_memory_usage = np.mean(full_memory_usages)
            avg_full_accuracy = np.mean(full_accuracies)
            avg_full_f1 = np.mean(full_f1_scores)
            
            avg_hybrid_train_time = np.mean(hybrid_train_times)
            avg_hybrid_inference_time = np.mean(hybrid_inference_times)
            avg_hybrid_memory_usage = np.mean(hybrid_memory_usages)
            avg_hybrid_accuracy = np.mean(hybrid_accuracies)
            avg_hybrid_f1 = np.mean(hybrid_f1_scores)
            
            # Calculate improvements
            train_time_improvement = ((avg_full_train_time - avg_hybrid_train_time) / avg_full_train_time) * 100
            inference_time_improvement = ((avg_full_inference_time - avg_hybrid_inference_time) / avg_full_inference_time) * 100
            memory_usage_improvement = ((avg_full_memory_usage - avg_hybrid_memory_usage) / avg_full_memory_usage) * 100
            
            # Store results
            results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Feature_Set": "Full",
                "Num_Features": X_full.shape[1],
                "Train_Time_Sec": avg_full_train_time,
                "Inference_Time_Sec": avg_full_inference_time,
                "Memory_Usage_MB": avg_full_memory_usage,
                "Accuracy": avg_full_accuracy,
                "F1_Score": avg_full_f1
            })
            
            results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Feature_Set": "Hybrid",
                "Num_Features": X_hybrid.shape[1],
                "Train_Time_Sec": avg_hybrid_train_time,
                "Inference_Time_Sec": avg_hybrid_inference_time,
                "Memory_Usage_MB": avg_hybrid_memory_usage,
                "Accuracy": avg_hybrid_accuracy,
                "F1_Score": avg_hybrid_f1
            })
            
            # Print improvement percentages
            print(f"  Performance improvements with hybrid features:")
            print(f"    Training time: {train_time_improvement:.2f}% faster")
            print(f"    Inference time: {inference_time_improvement:.2f}% faster")
            if avg_full_memory_usage > 0:
                print(f"    Memory usage: {memory_usage_improvement:.2f}% lower")
            print(f"    Accuracy: {(avg_hybrid_accuracy - avg_full_accuracy)*100:.2f}% change")
            print(f"    F1 Score: {(avg_hybrid_f1 - avg_full_f1)*100:.2f}% change")
    
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        continue

# Check if we have any results to save
if not results:
    print("\nNo valid results collected. Please check datasets and try again.")
    sys.exit(1)

# Save results to CSV
os.makedirs("computational_efficiency", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv("computational_efficiency/detailed_results.csv", index=False)

# Create summary table for easy comparison
summary = []
for dataset_name in dataset_pairs.keys():
    for model_name in models.keys():
        full_data = results_df[(results_df["Dataset"] == dataset_name) & 
                              (results_df["Model"] == model_name) & 
                              (results_df["Feature_Set"] == "Full")]
        
        hybrid_data = results_df[(results_df["Dataset"] == dataset_name) & 
                                (results_df["Model"] == model_name) & 
                                (results_df["Feature_Set"] == "Hybrid")]
        
        if not full_data.empty and not hybrid_data.empty:
            train_time_improvement = ((full_data["Train_Time_Sec"].values[0] - hybrid_data["Train_Time_Sec"].values[0]) / 
                                     full_data["Train_Time_Sec"].values[0]) * 100
            
            inference_time_improvement = ((full_data["Inference_Time_Sec"].values[0] - hybrid_data["Inference_Time_Sec"].values[0]) / 
                                         full_data["Inference_Time_Sec"].values[0]) * 100
            
            memory_improvement = 0
            if full_data["Memory_Usage_MB"].values[0] > 0:
                memory_improvement = ((full_data["Memory_Usage_MB"].values[0] - hybrid_data["Memory_Usage_MB"].values[0]) / 
                                     full_data["Memory_Usage_MB"].values[0]) * 100
            
            accuracy_change = hybrid_data["Accuracy"].values[0] - full_data["Accuracy"].values[0]
            f1_change = hybrid_data["F1_Score"].values[0] - full_data["F1_Score"].values[0]
            
            summary.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Full_Features": full_data["Num_Features"].values[0],
                "Hybrid_Features": hybrid_data["Num_Features"].values[0],
                "Reduction": f"{(1 - hybrid_data['Num_Features'].values[0]/full_data['Num_Features'].values[0])*100:.1f}%",
                "Train_Time_Improvement": f"{train_time_improvement:.2f}%",
                "Inference_Time_Improvement": f"{inference_time_improvement:.2f}%",
                "Memory_Usage_Reduction": f"{memory_improvement:.2f}%",
                "Accuracy_Change": f"{accuracy_change*100:.2f}%",
                "F1_Score_Change": f"{f1_change*100:.2f}%"
            })

# Save summary to CSV
summary_df = pd.DataFrame(summary)
summary_df.to_csv("computational_efficiency/improvement_summary.csv", index=False)

# Create visualization
plt.figure(figsize=(15, 10))

# Plot training time comparison
plt.subplot(2, 2, 1)
sns.barplot(data=results_df, x="Dataset", y="Train_Time_Sec", hue="Feature_Set")
plt.title("Training Time Comparison")
plt.ylabel("Time (seconds)")
plt.xticks(rotation=45)
plt.legend(title="Feature Set")

# Plot inference time comparison
plt.subplot(2, 2, 2)
sns.barplot(data=results_df, x="Dataset", y="Inference_Time_Sec", hue="Feature_Set")
plt.title("Inference Time Comparison")
plt.ylabel("Time (seconds)")
plt.xticks(rotation=45)
plt.legend(title="Feature Set")

# Only plot memory usage if psutil was available
if not all(results_df["Memory_Usage_MB"] == 0):
    # Plot memory usage comparison
    plt.subplot(2, 2, 3)
    sns.barplot(data=results_df, x="Dataset", y="Memory_Usage_MB", hue="Feature_Set")
    plt.title("Memory Usage Comparison")
    plt.ylabel("Memory (MB)")
    plt.xticks(rotation=45)
    plt.legend(title="Feature Set")

# Plot accuracy comparison
plt.subplot(2, 2, 4)
sns.barplot(data=results_df, x="Dataset", y="F1_Score", hue="Feature_Set")
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)
plt.legend(title="Feature Set")

plt.tight_layout()
plt.savefig("computational_efficiency/visual_comparison.png")
plt.close()

# Create a summary visualization
plt.figure(figsize=(14, 8))

# Extract average improvements, excluding memory if not measured
metrics_to_include = ["Training Time", "Inference Time"]
if not all(results_df["Memory_Usage_MB"] == 0):
    metrics_to_include.append("Memory Usage")
metrics_to_include.append("Feature Count")

avg_improvements = {}
for metric in metrics_to_include:
    if metric == "Training Time":
        avg_improvements[metric] = summary_df["Train_Time_Improvement"].apply(lambda x: float(x.strip("%"))).mean()
    elif metric == "Inference Time":
        avg_improvements[metric] = summary_df["Inference_Time_Improvement"].apply(lambda x: float(x.strip("%"))).mean()
    elif metric == "Memory Usage":
        avg_improvements[metric] = summary_df["Memory_Usage_Reduction"].apply(lambda x: float(x.strip("%"))).mean()
    elif metric == "Feature Count":
        avg_improvements[metric] = summary_df["Reduction"].apply(lambda x: float(x.strip("%"))).mean()

# Create bar chart of improvements
colors = ['#3274A1', '#E1812C', '#3A923A', '#C03D3E'][:len(avg_improvements)]
plt.bar(avg_improvements.keys(), avg_improvements.values(), color=colors)
plt.title("Average Efficiency Improvements with Hybrid Feature Selection")
plt.ylabel("Improvement (%)")
plt.ylim(0, max(avg_improvements.values()) * 1.2)

# Add value labels on bars
for i, (key, value) in enumerate(avg_improvements.items()):
    plt.text(i, value + 1, f"{value:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("computational_efficiency/average_improvements.png")
plt.close()

print("\nAnalysis complete. Results saved to computational_efficiency/ directory:")
print("- detailed_results.csv: Raw measurements")
print("- improvement_summary.csv: Summary of improvements by dataset and model")
print("- visual_comparison.png: Comparison of metrics across datasets")
print("- average_improvements.png: Average efficiency gains across all tests")

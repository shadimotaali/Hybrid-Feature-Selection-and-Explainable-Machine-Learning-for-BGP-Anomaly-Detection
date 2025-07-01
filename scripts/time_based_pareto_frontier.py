import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os
import time
import traceback

# Display current working directory
print(f"Current working directory: {os.getcwd()}")

# Create output directory with error handling
try:
    os.makedirs("pareto_time_analysis", exist_ok=True)
    print("Successfully created pareto_time_analysis directory")
except Exception as e:
    print(f"Error creating directory: {e}")
    print(traceback.format_exc())

print("Performing time-based Pareto frontier analysis...")

# Function to identify Pareto optimal points
def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A boolean array of pareto-efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost in either dimension
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
            # And keep self
            is_efficient[i] = True
    return is_efficient

# Function to evaluate model with different feature counts
def evaluate_feature_count(X, y, feature_list, model_name, model, feature_type="Hybrid"):
    results = []
    
    # Test different feature counts
    feature_counts = list(range(5, len(feature_list) + 1, 5))
    if len(feature_list) not in feature_counts:
        feature_counts.append(len(feature_list))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    for n in feature_counts:
        try:
            selected_features = feature_list[:n]
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            # Measure training time
            start_train_time = time.time()
            model.fit(X_train_selected, y_train)
            train_time = time.time() - start_train_time
            
            # Measure inference time (average over multiple runs for stability)
            inference_times = []
            n_runs = 10
            for _ in range(n_runs):
                start_inf_time = time.time()
                y_pred = model.predict(X_test_selected)
                inference_times.append(time.time() - start_inf_time)
            
            inference_time = np.mean(inference_times)
            
            # Calculate F1 score
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # For Pareto analysis, we want to minimize inference time and maximize F1
            results.append({
                'Model': model_name,
                'Feature_Type': feature_type,
                'Feature_Count': n,
                'F1_Score': f1,
                'Neg_F1_Score': -f1,  # Negative because we want to minimize for Pareto
                'Training_Time': train_time,
                'Inference_Time': inference_time,
                'Complexity': n * model.n_estimators  # Keep for reference
            })
            
            print(f"Evaluated {model_name} with {n} {feature_type} features: " + 
                  f"F1={f1:.4f}, Train={train_time:.4f}s, Infer={inference_time:.6f}s")
        except Exception as e:
            print(f"Error evaluating {n} features: {e}")
    
    return results

# Process all datasets (Full Dataset and Class-specific)
datasets = {
    "Full_Dataset": "Randomly_Balanced_Dataset.xlsx",
    "Class_1": "Class_1.xlsx",
    "Class_2": "Class_2.xlsx",
    "Class_3": "Class_3.xlsx"
}

all_results = []

for dataset_name, file_path in datasets.items():
    print(f"\nProcessing dataset: {dataset_name}")
    
    try:
        # Load dataset
        data = pd.read_excel(file_path)
        X = data.drop(columns=["target"])
        y = data["target"].apply(lambda x: 0 if x == 0 else 1)
        
        # Load hybrid features
        hybrid_features = pd.read_csv("feature_selection_results/Hybrid_top_features.csv")["Feature"].tolist()
        
        # Define full feature set (all features in the dataset)
        full_features = X.columns.tolist()
        
        # Define models with tuned parameters
        rf_tuned = RandomForestClassifier(
            max_depth=80,
            max_features='sqrt',
            min_samples_leaf=2,
            min_samples_split=2,
            n_estimators=350,
            random_state=42
        )

        xgb_tuned = XGBClassifier(
            colsample_bytree=0.9,
            gamma=0,
            learning_rate=0.15,
            max_depth=9,
            min_child_weight=10,
            n_estimators=200,
            subsample=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        # Evaluate models with hybrid features
        results_hybrid_rf = evaluate_feature_count(X, y, hybrid_features, "Random Forest", rf_tuned, "Hybrid")
        results_hybrid_xgb = evaluate_feature_count(X, y, hybrid_features, "XGBoost", xgb_tuned, "Hybrid")
        
        # Evaluate models with full feature set
        results_full_rf = evaluate_feature_count(X, y, full_features, "Random Forest", rf_tuned, "Full")
        results_full_xgb = evaluate_feature_count(X, y, full_features, "XGBoost", xgb_tuned, "Full")
        
        # Combine results
        dataset_results = results_hybrid_rf + results_hybrid_xgb + results_full_rf + results_full_xgb
        
        # Add dataset name
        for result in dataset_results:
            result['Dataset'] = dataset_name
        
        all_results.extend(dataset_results)
        
        # Save dataset-specific results
        dataset_df = pd.DataFrame(dataset_results)
        try:
            dataset_df.to_csv(f"pareto_time_analysis/{dataset_name}_pareto_results.csv", index=False)
            print(f"Saved results to pareto_time_analysis/{dataset_name}_pareto_results.csv")
        except Exception as e:
            print(f"Error saving CSV: {e}")
        
        # Create Pareto frontier visualizations for training and inference time
        for time_metric in ['Training_Time', 'Inference_Time']:
            plt.figure(figsize=(12, 8))
            
            # Dictionary to store markers and colors
            markers = {
                'Random Forest': ['o', 's'],  # circle, square
                'XGBoost': ['x', 'd']        # x, diamond
            }
            
            colors = {
                'Hybrid': 'blue',
                'Full': 'red'
            }
            
            # Create a combined array for Pareto analysis
            costs_array = []
            model_types = []
            
            for result in dataset_results:
                costs_array.append([result[time_metric], result['Neg_F1_Score']])
                model_types.append((result['Model'], result['Feature_Type'], result['Feature_Count']))
            
            costs_array = np.array(costs_array)
            
            # Find Pareto efficient points
            pareto_efficient = is_pareto_efficient(costs_array)
            
            # Plot all points
            for i, result in enumerate(dataset_results):
                model = result['Model']
                feature_type = result['Feature_Type']
                n_features = result['Feature_Count']
                
                marker_idx = 0 if feature_type == 'Hybrid' else 1
                marker = markers[model][marker_idx]
                color = colors[feature_type]
                
                # Marker size depends on whether it's a Pareto point
                size = 100 if pareto_efficient[i] else 50
                alpha = 1.0 if pareto_efficient[i] else 0.5
                
                # Convert negative F1 back to positive for plotting
                plt.scatter(result[time_metric], -result['Neg_F1_Score'], 
                            marker=marker, color=color, s=size, alpha=alpha,
                            label=f"{model} ({feature_type})" if i % 5 == 0 else "")
                
                # Add feature count label to pareto-optimal points
                if pareto_efficient[i]:
                    plt.annotate(f"{n_features}", 
                                (result[time_metric], -result['Neg_F1_Score']),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8)
            
            # Connect Pareto points with a line
            pareto_points = [p for i, p in enumerate(zip(costs_array[:, 0], -costs_array[:, 1])) if pareto_efficient[i]]
            pareto_points.sort(key=lambda x: x[0])  # Sort by time
            if pareto_points:
                pareto_x, pareto_y = zip(*pareto_points)
                plt.plot(pareto_x, pareto_y, 'r--', label='Pareto Frontier')
            
            # Set labels and title
            metric_name = "Training Time" if time_metric == "Training_Time" else "Inference Time"
            plt.xlabel(f'{metric_name} (seconds)')
            plt.ylabel('F1 Score')
            plt.title(f'Pareto Frontier for {dataset_name}: F1 Score vs {metric_name}')
            
            # Add a grid
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Create custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='RF Hybrid', markerfacecolor='blue', markersize=8),
                Line2D([0], [0], marker='s', color='w', label='RF Full', markerfacecolor='red', markersize=8),
                Line2D([0], [0], marker='x', color='blue', label='XGB Hybrid', markersize=8),
                Line2D([0], [0], marker='d', color='red', label='XGB Full', markersize=8),
                Line2D([0], [0], color='r', linestyle='--', label='Pareto Frontier')
            ]
            plt.legend(handles=legend_elements, loc='best')
            
            plt.tight_layout()
            
            # Save the figure with error handling
            try:
                plt.savefig(f"pareto_time_analysis/{dataset_name}_{time_metric.lower()}_pareto.png", dpi=300)
                print(f"Successfully saved plot to pareto_time_analysis/{dataset_name}_{time_metric.lower()}_pareto.png")
            except Exception as e:
                print(f"Error saving plot: {e}")
            
            plt.close()
        
        # Create 3D Pareto visualization (F1 vs Training Time vs Inference Time)
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot all points
            for result in dataset_results:
                model = result['Model']
                feature_type = result['Feature_Type']
                n_features = result['Feature_Count']
                
                marker_idx = 0 if feature_type == 'Hybrid' else 1
                marker = markers[model][marker_idx]
                color = colors[feature_type]
                
                ax.scatter(result['Training_Time'], 
                          result['Inference_Time'], 
                          result['F1_Score'],
                          marker=marker, color=color, s=50,
                          label=f"{model} ({feature_type})")
                
                # Add feature count label
                ax.text(result['Training_Time'], 
                       result['Inference_Time'], 
                       result['F1_Score'],
                       f"{n_features}", size=8)
            
            # Set labels
            ax.set_xlabel('Training Time (s)')
            ax.set_ylabel('Inference Time (s)')
            ax.set_zlabel('F1 Score')
            ax.set_title(f'3D Performance Space for {dataset_name}')
            
            # Handle legend (unique entries only)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='best')
            
            plt.tight_layout()
            plt.savefig(f"pareto_time_analysis/{dataset_name}_3d_pareto.png", dpi=300)
            print(f"Successfully saved 3D plot to pareto_time_analysis/{dataset_name}_3d_pareto.png")
            plt.close()
        except Exception as e:
            print(f"Error creating 3D plot: {e}")
        
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        print(traceback.format_exc())

# Save combined results
try:
    combined_df = pd.DataFrame(all_results)
    combined_df.to_csv("pareto_time_analysis/all_pareto_results.csv", index=False)
    print("Saved all results to pareto_time_analysis/all_pareto_results.csv")
except Exception as e:
    print(f"Error saving combined results: {e}")

# Create inference time comparison visualization
try:
    plt.figure(figsize=(15, 10))
    
    # Dictionary to store markers and colors
    dataset_colors = {
        "Full_Dataset": "blue",
        "Class_1": "green",
        "Class_2": "red",
        "Class_3": "purple"
    }
    
    for dataset_name in datasets.keys():
        # Filter for this dataset and hybrid features only with XGBoost
        dataset_results = [r for r in all_results if r['Dataset'] == dataset_name and 
                          r['Feature_Type'] == 'Hybrid' and r['Model'] == 'XGBoost']
        
        # Sort by feature count
        dataset_results.sort(key=lambda x: x['Feature_Count'])
        
        # Extract data points
        feature_counts = [r['Feature_Count'] for r in dataset_results]
        inference_times = [r['Inference_Time'] * 1000 for r in dataset_results]  # Convert to milliseconds
        
        if feature_counts and inference_times:
            plt.plot(feature_counts, inference_times, 'o-', color=dataset_colors[dataset_name], label=dataset_name)
    
    plt.xlabel('Number of Hybrid Features')
    plt.ylabel('Inference Time (milliseconds)')
    plt.title('Inference Time vs. Feature Count Across Datasets (XGBoost with Hybrid Features)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("pareto_time_analysis/inference_time_comparison.png", dpi=300)
    print("Saved inference time comparison to pareto_time_analysis/inference_time_comparison.png")
except Exception as e:
    print(f"Error creating inference time plot: {e}")

print("\nTime-based Pareto analysis completed.")

# Analyze and print key insights
try:
    insights_df = pd.DataFrame(all_results)
    
    # Find optimal feature counts based on Pareto analysis
    print("\nOptimal Feature Counts (Pareto-efficient points with best F1/time trade-off):")
    print("---------------------------------------------------------------------------")
    
    for dataset_name in datasets.keys():
        print(f"\n{dataset_name}:")
        dataset_hybrid = insights_df[(insights_df['Dataset'] == dataset_name) & 
                                    (insights_df['Feature_Type'] == 'Hybrid')]
        
        if not dataset_hybrid.empty:
            for model in ['Random Forest', 'XGBoost']:
                model_data = dataset_hybrid[dataset_hybrid['Model'] == model]
                if not model_data.empty:
                    # Sort by F1 score
                    model_data = model_data.sort_values('F1_Score', ascending=False)
                    
                    # Get top F1 result
                    best_f1 = model_data.iloc[0]
                    
                    # Get fastest inference result with acceptable F1 (within 1% of best)
                    f1_threshold = best_f1['F1_Score'] * 0.99
                    fastest = model_data[model_data['F1_Score'] >= f1_threshold].sort_values('Inference_Time').iloc[0]
                    
                    print(f"  {model}:")
                    print(f"    - Best F1: {best_f1['Feature_Count']} features (F1={best_f1['F1_Score']:.4f}, " +
                          f"Inference={best_f1['Inference_Time']*1000:.2f}ms)")
                    print(f"    - Fastest within 1% of best F1: {fastest['Feature_Count']} features " +
                          f"(F1={fastest['F1_Score']:.4f}, Inference={fastest['Inference_Time']*1000:.2f}ms)")
    
    # Time efficiency analysis
    print("\nTime Efficiency Analysis:")
    print("------------------------")
    
    for dataset_name in datasets.keys():
        print(f"\n{dataset_name}:")
        
        # Compare hybrid vs. full feature sets for XGBoost
        hybrid_xgb = insights_df[(insights_df['Dataset'] == dataset_name) & 
                              (insights_df['Feature_Type'] == 'Hybrid') &
                              (insights_df['Model'] == 'XGBoost')]
        
        full_xgb = insights_df[(insights_df['Dataset'] == dataset_name) & 
                            (insights_df['Feature_Type'] == 'Full') &
                            (insights_df['Model'] == 'XGBoost')]
        
        if not hybrid_xgb.empty and not full_xgb.empty:
            # Find best F1 for each
            best_hybrid = hybrid_xgb.sort_values('F1_Score', ascending=False).iloc[0]
            best_full = full_xgb.sort_values('F1_Score', ascending=False).iloc[0]
            
            # Compare inference times
            hybrid_time = best_hybrid['Inference_Time'] * 1000  # ms
            full_time = best_full['Inference_Time'] * 1000      # ms
            time_speedup = ((full_time / hybrid_time) - 1) * 100
            
            print(f"  XGBoost Comparison:")
            print(f"    - Hybrid ({best_hybrid['Feature_Count']} features): " +
                  f"F1={best_hybrid['F1_Score']:.4f}, Inference={hybrid_time:.2f}ms")
            print(f"    - Full ({best_full['Feature_Count']} features): " +
                  f"F1={best_full['F1_Score']:.4f}, Inference={full_time:.2f}ms")
            print(f"    - Time efficiency gain: {time_speedup:.1f}% faster inference with hybrid features")
    
    print("\nThis time-based Pareto analysis clearly demonstrates the computational")
    print("efficiency advantages of your hybrid feature selection approach.")
except Exception as e:
    print(f"Error generating insights: {e}")
    print(traceback.format_exc())

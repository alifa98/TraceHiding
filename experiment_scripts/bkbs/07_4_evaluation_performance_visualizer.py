import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Create output directory for plots
os.makedirs("plots", exist_ok=True)

# Define constants
DATASET_NAME = "Ho_Foursquare_NYC"
MODEL_NAME = "LSTM"
BASELINE_METHODS = ["original", "retraining", "our_method", "finetune", "neg_grad", "neg_grad_plus", "bad-t", "scrub"]  # Add more methods as needed
RANDOM_SAMPLE_UNLEARNING_SIZES = [200]
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

# Initialize data storage for results
results = []

# Load data
for method in BASELINE_METHODS:
    for sample_size in RANDOM_SAMPLE_UNLEARNING_SIZES:
        # Prepare to collect repetitions of metrics for this configuration
        unlearning_dataset_metrics = []
        test_dataset_metrics = []
        
        for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
            save_path = f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/{method}/performance_metrics.json"
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    performance_metrics = json.load(f)
                
                # Append metrics for both datasets (Unlearning & Test)
                unlearning_dataset_metrics.append(performance_metrics["performance for Unlearning Dataset"])
                test_dataset_metrics.append(performance_metrics["performance for Test Dataset"])
            else:
                print(f"File {save_path} not found for {method}, skipping.")
        
        # If no data was collected, skip this method
        if not unlearning_dataset_metrics or not test_dataset_metrics:
            print(f"No valid data for method {method} and sample size {sample_size}. Skipping this method.")
            continue
        
        # Convert to DataFrame for easier calculation
        df_unlearning = pd.DataFrame(unlearning_dataset_metrics)
        df_test = pd.DataFrame(test_dataset_metrics)

        # Function to calculate mean and confidence intervals
        def calculate_stats(df):
            if len(df) > 1:  # Ensure we have more than 1 sample to calculate CI
                mean = df.mean()
                ci = df.apply(lambda x: stats.sem(x) * stats.t.ppf(0.975, len(x) - 1))  # 95% confidence interval
            else:
                mean = df.mean()
                ci = pd.Series([0] * len(mean), index=mean.index)  # Set CI to 0 if not enough data
            return mean, ci

        unlearning_mean, unlearning_ci = calculate_stats(df_unlearning)
        test_mean, test_ci = calculate_stats(df_test)

        # Store the results for tabular display
        results.append({
            'Method': method,
            'Sample Size': sample_size,
            'Dataset': 'Unlearning',
            'Metrics': unlearning_mean.to_dict(),
            'CI': unlearning_ci.to_dict()
        })

        results.append({
            'Method': method,
            'Sample Size': sample_size,
            'Dataset': 'Test',
            'Metrics': test_mean.to_dict(),
            'CI': test_ci.to_dict()
        })

# Save the table of metrics with confidence intervals as a CSV file
result_df = pd.json_normalize(results)
result_df.to_csv(f"plots/{DATASET_NAME}_performance_metrics.csv", index=False)
print(f"Performance metrics table saved as {DATASET_NAME}_performance_metrics.csv")

# Create grouped bar plots with error bars for each dataset and method
def plot_grouped_performance(results, dataset_name):
    # Filter the results for the specific dataset (Unlearning or Test)
    dataset_results = [r for r in results if r['Dataset'] == dataset_name]

    if len(dataset_results) == 0:
        print(f"No data available for {dataset_name} dataset.")
        return

    # Extract the list of metrics
    metrics = list(dataset_results[0]['Metrics'].keys())

    # Number of methods
    n_methods = len(dataset_results)

    # Set up bar chart properties
    x = np.arange(len(metrics))  # Label locations
    width = 0.2  # Width of the bars for each method
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Loop through each method's results and plot the bars
    for i, result in enumerate(dataset_results):
        metrics_mean = pd.Series(result['Metrics'])
        metrics_ci = pd.Series(result['CI'])
        method = result['Method']

        # Ensure there are matching shapes between metrics and CIs
        if len(metrics_mean) != len(metrics_ci):
            print(f"Skipping method {method} due to shape mismatch between metrics and CIs.")
            continue

        # Shift the bar positions for each method
        ax.bar(x + i * width, metrics_mean.values, width, yerr=metrics_ci.values, capsize=5, label=method)

    # Add labels and titles
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title(f'Performance Metrics for {dataset_name}')
    ax.set_xticks(x + width * (n_methods - 1) / 2)  # Adjust for centering the groups
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend(title='Method')

    plt.tight_layout()

    # Save the plot as an image
    plot_filename = f"plots/{DATASET_NAME}_{dataset_name}_performance_metrics_grouped.png"
    plt.savefig(plot_filename)
    print(f"Grouped plot saved as {plot_filename}")

    plt.close()  # Close the plot to avoid showing in non-interactive environments

# Plot the grouped bar charts for Unlearning and Test datasets
plot_grouped_performance(results, 'Unlearning')
plot_grouped_performance(results, 'Test')

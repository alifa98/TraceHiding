import json
import numpy as np
from scipy import stats

DATASET = "Ho_Foursquare_NYC"
MODEL = "LSTM"
SAMPLE_SIZE = 20
METHODS = ["original", "retraining", "finetune", "neg_grad", "neg_grad_plus", "bad-t", "scrub", "trace_hiding"] #  add "trace_hiding"
IMPORTANCE = "entropy"
EPOCH = 14
MIA = "xgboost"
NUM_RUNS = 5

metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]

all_data_results = {}

for method_name in METHODS:

    experiment_data = []
    for i in range(NUM_RUNS):
        
        if method_name == "original":
            results_json_path = f"experiments/{DATASET}/saved_models/{MODEL}/evaluation/metrics_{MIA}_mia_epoch_{EPOCH}.json"
        else:
            results_json_path = f"experiments/{DATASET}/unlearning/user_sample/sample_size_{SAMPLE_SIZE}/sample_{i}/{MODEL}/{method_name}/{IMPORTANCE +'/' if method_name == 'trace_hiding' else ''}evaluation/metrics_{MIA}_mia_epoch_{EPOCH}.json"
        
        with open(results_json_path, "r") as f:
            experiment_data.append(json.load(f))

    results = {metric: np.array([exp[metric] for exp in experiment_data]) for metric in metrics}

    # Confidence level (95%)
    confidence_level = 0.95
    ci_results = {}

    # Calculate confidence intervals using t-distribution for each metric
    for metric, values in results.items():
        mean = np.mean(values)
        n = len(values)
        std_dev = np.std(values, ddof=1)  # Sample standard deviation
        standard_error = std_dev / np.sqrt(n)  # Standard error of the mean
        
        # t-interval for the confidence level
        ci_lower, ci_upper = stats.t.interval(confidence_level, n - 1, loc=mean, scale=standard_error)
        
        ci_results[metric] = {
            "mean": mean,
            "confidence_interval": (ci_lower, ci_upper),
            "plus_minus": f"{mean*100:.2f} ± {(ci_upper - mean)*100:.2f}"
        }
        
        all_data_results[method_name] = ci_results


for metric in metrics:
    for method in METHODS:
        # print(f"{method} {metric} {all_data_results[method][metric]['mean']*100:.2f}")
        print(f"{all_data_results[method][metric]['mean']*100:.2f}", end=' & ')
    print()
    
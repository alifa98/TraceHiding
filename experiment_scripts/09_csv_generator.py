import csv
import json
import os
import sys
import logging
import glob
import os

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utility.functions import compute_confidence_interval

datasets = [
    {"name": "HO_Rome_Res8", "sample_sizes": [2, 10, 21, 43]},
    {"name": "HO_Geolife_Res8", "sample_sizes": [1, 2, 3, 5]},
    {"name": "HO_NYC_Res9", "sample_sizes": [2, 11, 23, 46]},
]

sample_index_to_percentage = {0: "1", 1: "5", 2: "10", 3: "20"}

models = ["GRU", "LSTM", "BERT", "ModernBERT"]

BIASED_SAMPLE_IMPORTANCE_NAME = None # "entropy_max" or None
METHODS_EPOCH = {
    "retraining": None,
    "finetune": None,
    "neg_grad": None,
    "neg_grad_plus": None,
    "bad-t": None,
    "scrub": None,
    "trace_hiding": None,
}
trace_hiding_importances = ["entropy", "coverage_diversity", "unified"]
MIA = "lr"
SCENARIO = "user"
NUM_RUNS = 5

showing_name_for_table = {
    "retraining": "Retraining",
    "finetune": "Finetuning",
    "neg_grad": "NegGrad",
    "neg_grad_plus": "NegGrad+",
    "bad-t": "Bad-T",
    "scrub": "SCRUB",
    "trace_hiding": "TraceHiding",  # importance will be included per-row
}

csv_rows = []
header = [
    "dataset",
    "model",
    "method",
    "importance",  # Only filled for trace_hiding; others can be empty
    "sample_percent",  # Percent of the dataset used for the sample
    "UA_mean",
    "UA_pm",
    "UA_n",
    "RA_mean",
    "RA_pm",
    "RA_n",
    "TA_mean",
    "TA_pm",
    "TA_n",
    "MIA_delta_mean",
    "MIA_delta_pm",
    "MIA_n",
    "Time",  # Time taken for the experiments
    "Time_pm",
    "Time_n",  # Number of runs for the time
]


# Compute means, CIs, and N
def get_mean_pm_n(values):
    valid_values = [v for v in values if v != -1]
    if valid_values:
        mean, pm = compute_confidence_interval(valid_values)
        n = len(valid_values)
    else:
        mean, pm, n = -1, 0.0, 0
    return mean, pm, n


def find_unlearning_stats_json(directory):
    """
    Finds the JSON file matching the pattern 'unlearning_stats-batch_size_*.json'
    in the given directory and returns its full path.

    Args:
        directory (str): Path to the directory to search in.

    Returns:
        str or None: Full path to the matched JSON file, or None if not found.
    """
    pattern = os.path.join(directory, "unlearning_stats-batch_size_*.json")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


for dataset in datasets:
    DATASET_NAME = dataset["name"]
    SAMPLE_SIZES = dataset["sample_sizes"]

    for model_name in models:
        for method_name, eval_epoch in METHODS_EPOCH.items():
            # For trace_hiding, iterate over importances; otherwise, just once
            importances = trace_hiding_importances if method_name == "trace_hiding" else [""]
            for importance in importances:
                for index, sample_size in enumerate(SAMPLE_SIZES):
                    # Convert sample size to percentage
                    sample_percent = sample_index_to_percentage[index]

                    experiment_data = []
                    for i in range(NUM_RUNS):
                        results_folder = (
                            f"experiments/{DATASET_NAME}/unlearning/"
                            f"{SCENARIO}_sample{f'_biased_{BIASED_SAMPLE_IMPORTANCE_NAME}' if BIASED_SAMPLE_IMPORTANCE_NAME else ''}/"
                            f"sample_size_{sample_size}/sample_{i}/"
                            f"{model_name}/{method_name}/"
                            f"{importance if method_name == 'trace_hiding' else ''}"
                        )
                        mia_results_json_path = f"{results_folder}/evaluation/metrics_{MIA}_mia_epoch_{eval_epoch}.json"
                        perf_results_json_path = f"{results_folder}/evaluation/metrics_performance_epoch_{eval_epoch}.json"
                        stats_results_json_path = find_unlearning_stats_json(results_folder) if method_name != "retraining" else f"{results_folder}/retrained_{model_name}_model.json"
                        data1 = {"mia": {"auc_roc": -1}, "time": -1}
                        data2 = {
                            "unlearning_dataset": {"accuracy@1": -1},
                            "remaining_dataset":  {"accuracy@1": -1},
                            "test_dataset":       {"accuracy@1": -1},
                        }
                        try:
                            with open(mia_results_json_path, "r") as f:
                                data1 = {"mia": json.load(f)}
                        except (FileNotFoundError, json.JSONDecodeError):
                            logging.warning(f"File not found or invalid JSON: {mia_results_json_path}")
                        
                        
                        try:
                            with open(perf_results_json_path, "r") as f:
                                data2 = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError):
                            logging.warning(f"File not found or invalid JSON: {perf_results_json_path}")
                        
                        try:
                            if stats_results_json_path and method_name != "retraining":
                                with open(stats_results_json_path, "r") as f:
                                    time = 0
                                    for epoch, stat_dict in json.load(f).items():
                                        time += stat_dict["epoch_time"]
                                    data1["time"] = time
                            elif stats_results_json_path and method_name == "retraining":
                                with open(stats_results_json_path, "r") as f:
                                    stat_dict = json.load(f)
                                    data1["time"] = stat_dict["training_time"]
                         
                        except (FileNotFoundError, json.JSONDecodeError):
                            logging.warning(f"File not found or invalid JSON: {stats_results_json_path}")
                        
                        experiment_data.append({**data1, **data2})

                    # Prepare arrays for the desired metrics
                    results = {"UA": [], "RA": [], "TA": [], "MIA_AUC": [], "Time" : []}

                    for i in range(NUM_RUNS):
                        unlearning_acc = experiment_data[i]["unlearning_dataset"][
                            "accuracy@1"
                        ]
                        remaining_acc = experiment_data[i]["remaining_dataset"][
                            "accuracy@1"
                        ]
                        test_acc = experiment_data[i]["test_dataset"]["accuracy@1"]
                        mia_auc = experiment_data[i]["mia"]["auc_roc"]
                        UA_value = -1 if unlearning_acc == -1 else (1 - unlearning_acc)
                        results["UA"].append(UA_value)
                        results["RA"].append(remaining_acc)
                        results["TA"].append(test_acc)
                        results["MIA_AUC"].append(mia_auc)
                        results["Time"].append(experiment_data[i]["time"])

                    UA_mean, UA_pm, UA_n = get_mean_pm_n(results["UA"])
                    RA_mean, RA_pm, RA_n = get_mean_pm_n(results["RA"])
                    TA_mean, TA_pm, TA_n = get_mean_pm_n(results["TA"])
                    MIA_mean, MIA_pm, MIA_n = get_mean_pm_n(results["MIA_AUC"])
                    Time_mean, Time_pm, Time_n = get_mean_pm_n(results["Time"])
                    if MIA_mean == -1:
                        MIA_delta_mean, MIA_delta_pm = -1, 0.0
                    else:
                        MIA_delta_mean = abs(MIA_mean - 0.5) # Difference from random classifier
                        MIA_delta_pm = MIA_pm

                    row = [
                        DATASET_NAME,
                        model_name,
                        showing_name_for_table[method_name],
                        importance if method_name == "trace_hiding" else "",
                        sample_percent,
                        UA_mean,
                        UA_pm,
                        UA_n,
                        RA_mean,
                        RA_pm,
                        RA_n,
                        TA_mean,
                        TA_pm,
                        TA_n,
                        MIA_delta_mean,
                        MIA_delta_pm,
                        MIA_n,
                        Time_mean,
                        Time_pm,
                        Time_n
                    ]
                    csv_rows.append(row)

# Now write all rows to CSV
csv_filename = f"all_results_{BIASED_SAMPLE_IMPORTANCE_NAME if BIASED_SAMPLE_IMPORTANCE_NAME else "uniform" }_sampling.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(csv_rows)

print(f"All CSV results written to {csv_filename}")

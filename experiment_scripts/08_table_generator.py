import json
import os
import sys
import numpy as np
from scipy import stats
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utility.functions import compute_confidence_interval

# DATASET_NAME = "HO_Rome_Res8"
# SAMPLE_SIZES = [2, 10, 21, 43]

# DATASET_NAME = "HO_Geolife_Res8"
# SAMPLE_SIZES = [1, 2, 3, 5] 

DATASET_NAME = "HO_NYC_Res9"
SAMPLE_SIZES = [2, 11, 23, 46]

# DATASET_NAME = "HO_Porto_Res8"
# SAMPLE_SIZES = [4, 21, 43, 88]


BIASED_SAMPLE_IMPORTANCE_NAME = "entropy_max"
MODEL_NAME = "GRU"
METHODS_EPOCH = {
    "retraining": None,  # The epoch number for retraining is not important
    "finetune": None,
    "neg_grad": None,
    "neg_grad_plus": None,
    "bad-t": None,
    "scrub": None,
    "trace_hiding": None,
}
IMPORTANCE = "coverage_diversity"
# IMPORTANCE = "uuniqe"
# IMPORTANCE = "entropy"
MIA = "xgboost"
SCENARIO = "user"
NUM_RUNS = 5
INCLUDE_CONFIDENCE_INTERVALS = False

showing_name_for_table = {
    "retraining": "Retraining",
    "finetune": "Finetuning",
    "neg_grad": "NegGrad",
    "neg_grad_plus": "NegGrad+",
    "bad-t": "Bad-T",
    "scrub": "SCRUB",
    "trace_hiding": f"TraceHiding({IMPORTANCE})",
}

for method_name, epoch in METHODS_EPOCH.items():
    print(f"{showing_name_for_table[method_name]}", end=" & ")
    for sample_size in SAMPLE_SIZES:
        experiment_data = []
        for i in range(NUM_RUNS):
            
            results_folder = f"experiments/{DATASET_NAME}/unlearning/{SCENARIO}_sample{f"_biased_{BIASED_SAMPLE_IMPORTANCE_NAME}" if BIASED_SAMPLE_IMPORTANCE_NAME else ""}/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/{method_name}/{IMPORTANCE +'/' if method_name == 'trace_hiding' else ''}evaluation"
            
            mia_results_json_path = f"{results_folder}/metrics_{MIA}_mia_epoch_{epoch}.json"
            perf_results_json_path = f"{results_folder}/metrics_performance_epoch_{epoch}.json"
            
            # Initialize placeholders with -1 in case file reading fails
            data1 = {"mia": {"auc_roc": -1}}
            data2 = {
                "unlearning_dataset": {"accuracy@1": -1},
                "remaining_dataset":  {"accuracy@1": -1},
                "test_dataset":       {"accuracy@1": -1},
            }
            
            # Attempt to read MIA file
            try:
                with open(mia_results_json_path, "r") as f:
                    data1 = {"mia": json.load(f)}
            except (FileNotFoundError, json.JSONDecodeError) as e:
                # Could not read MIA file, keep the default -1
                pass
            
            # Attempt to read Performance file
            try:
                with open(perf_results_json_path, "r") as f:
                    data2 = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                # Could not read Performance file, keep the default -1
                pass

            # Combine results
            experiment_data.append({**data1, **data2})
        
        # Prepare arrays for the desired metrics
        results = {
            "UA": [],
            "RA": [],
            "TA": [],
            "MIA_AUC": []
        }

        # Extract the metrics (put -1 where the file was not found)
        for i in range(NUM_RUNS):
            unlearning_acc  = experiment_data[i]["unlearning_dataset"]["accuracy@1"]
            remaining_acc   = experiment_data[i]["remaining_dataset"]["accuracy@1"]
            test_acc        = experiment_data[i]["test_dataset"]["accuracy@1"]
            mia_auc         = experiment_data[i]["mia"]["auc_roc"]

            # If any of them is -1, keep it that way or handle carefully:
            # For UA (1 - unlearning_acc), avoid doing (1 - -1). Instead store -1 if not found.
            UA_value = -1 if unlearning_acc == -1 else (1 - unlearning_acc)

            results["UA"].append(UA_value)
            results["RA"].append(remaining_acc)
            results["TA"].append(test_acc)
            results["MIA_AUC"].append(mia_auc)

        # Compute means & confidence intervals, skipping -1 runs
        ci_results = {}
        for metric, values in results.items():
            # Filter out any -1
            valid_values = [v for v in values if v != -1]
            
            if len(valid_values) > 0:
                mean, pm = compute_confidence_interval(valid_values)
            else:
                # If all runs are -1, mark the final result as -1
                mean, pm = -1, 0.0
            
            ci_results[metric] = {
                "mean": mean,
                "plus_minus": pm
            }
        
        # Print the result line for this sample_size
        # The printing structure is unchanged, but note that if 'mean' is -1,
        # we simply display -1
        def fmt_value(m):
            """Helper to format metric mean (e.g., 25.34) or -1 as needed."""
            if m == -1:
                return "-1"
            return f"{m*100:.2f}".rstrip('0').rstrip('.')

        # UA
        print(fmt_value(ci_results['UA']['mean']), end=" ")
        if INCLUDE_CONFIDENCE_INTERVALS and ci_results['UA']['mean'] != -1:
            print(f"\\pm {(ci_results['UA']['plus_minus'])*100:.2f}".rstrip('0').rstrip('.'), end=" ")
        print(" & ", end=" ")

        # RA
        print(fmt_value(ci_results['RA']['mean']), end=" ")
        if INCLUDE_CONFIDENCE_INTERVALS and ci_results['RA']['mean'] != -1:
            print(f"\\pm {(ci_results['RA']['plus_minus'])*100:.2f}".rstrip('0').rstrip('.'), end=" ")
        print(" & ", end=" ")

        # TA
        print(fmt_value(ci_results['TA']['mean']), end=" ")
        if INCLUDE_CONFIDENCE_INTERVALS and ci_results['TA']['mean'] != -1:
            print(f"\\pm {(ci_results['TA']['plus_minus'])*100:.2f}".rstrip('0').rstrip('.'), end=" ")
        print(" & ", end=" ")

        # MIA_AUC
        # The difference from 0.5 does not make sense if -1, so only compute if not -1
        mia_mean = ci_results['MIA_AUC']['mean']
        if mia_mean == -1:
            print("-1", end=" ")
        else:
            # Print the difference from 0.5
            mia_delta = abs(mia_mean - 0.5) * 100
            print(f"{mia_delta:.2f}".rstrip('0').rstrip('.'), end=" ")
            if INCLUDE_CONFIDENCE_INTERVALS:
                print(f"\\pm {ci_results['MIA_AUC']['plus_minus']:.2f}".rstrip('0').rstrip('.'), end=" ")

        if sample_size != SAMPLE_SIZES[-1]:
            print(" & ", end=" ")
    print("\\\\")

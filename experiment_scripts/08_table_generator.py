import json
import os
import sys
import numpy as np
from scipy import stats
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utility.functions import compute_confidence_interval

DATASET_NAME = "HO_Rome_Res8"
MODEL_NAME = "LSTM"
SAMPLE_SIZES = [1, 5, 10, 20]
METHODS_EPOCH = {
    "retraining": 14,
    # "finetune": 14,
    # "neg_grad": 14,
    # "neg_grad_plus": 14,
    # "bad-t": 14,
    # "scrub": 14,
    # "trace_hiding": 14,
}
IMPORTANCE = "entropy"
MIA = "xgboost"
SCENARIO = "user"
NUM_RUNS = 5
INCLUDE_CONFIDENCE_INTERVALS = False

for method_name, epoch in METHODS_EPOCH.items():
    print(f"{method_name}", end=" & ")
    for sample_size in SAMPLE_SIZES:
        experiment_data = []
        for i in range(NUM_RUNS):
            
            results_folder = f"experiments/{DATASET_NAME}/unlearning/{SCENARIO}_sample/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/{method_name}/{IMPORTANCE +'/' if method_name == 'trace_hiding' else ''}evaluation"
            
            mia_results_json_path = f"{results_folder}/metrics_{MIA}_mia_epoch_{epoch}.json"
            perf_results_json_path = f"{results_folder}/metrics_performance_epoch_{epoch}.json"
            
            with open(mia_results_json_path, "r") as f:
                data1 = {"mia":json.load(f)}
            with open(perf_results_json_path, "r") as f:
                data2 = json.load(f)
                
            experiment_data.append({**data1, **data2})
        
        results = {
            "UA": [],
            "RA": [],
            "TA": [],
            "MIA_AUC": []
        }
        # select the metrics of interest
        for i in range(NUM_RUNS):
            results["UA"].append(1 - experiment_data[i]["unlearning_dataset"]["accuracy@1"]) # define the unlearning accuracy
            results["RA"].append(experiment_data[i]["remaining_dataset"]["accuracy@1"])
            results["TA"].append(experiment_data[i]["test_dataset"]["accuracy@1"])
            results["MIA_AUC"].append(experiment_data[i]["mia"]["auc_roc"])

        ci_results = {}
        # Calculate confidence intervals using t-distribution for each metric
        for metric, values in results.items():
            mean, pm = compute_confidence_interval(values)
            ci_results[metric] = {
                "mean": mean,
                "plus_minus": pm
            }
            
        # Print the results / the order is UA, RA, TA, MIA_AUC
        # order of sample size is defined in the list SAMPLE_SIZES
        # order of methods is defined in the dictionary METHODS_EPOCH
        
        print(f"{ci_results['UA']['mean']*100:.2f}".rstrip('0').rstrip('.'), end=" ")
        if INCLUDE_CONFIDENCE_INTERVALS:
            print(f"\\pm {(ci_results['UA']['plus_minus'])*100:.2f}".rstrip('0').rstrip('.'), end=" ")
        print(" & ", end=" ")
        
        print(f"{ci_results['RA']['mean']*100:.2f}".rstrip('0').rstrip('.'), end=" ")
        if INCLUDE_CONFIDENCE_INTERVALS:
            print(f"\\pm {(ci_results['RA']['plus_minus'])*100:.2f}".rstrip('0').rstrip('.'), end=" ")
        print(" & ", end=" ")
        
        print(f"{ci_results['TA']['mean']*100:.2f}".rstrip('0').rstrip('.'), end=" ")
        if INCLUDE_CONFIDENCE_INTERVALS:
            print(f"\\pm {(ci_results['TA']['plus_minus'])*100:.2f}".rstrip('0').rstrip('.'), end=" ")
        print(" & ", end=" ")
        
        print(f"{ci_results['MIA_AUC']['mean']:.2f}".rstrip('0').rstrip('.'), end=" ")
        if INCLUDE_CONFIDENCE_INTERVALS:
            print(f"\\pm {(ci_results['MIA_AUC']['plus_minus']):.2f}".rstrip('0').rstrip('.'), end=" ")
            
        # if not the last sample size, print the separator (end of line in latex)
        if sample_size != SAMPLE_SIZES[-1]:
            print(" & ", end=" ")
            
    print("\\\\")
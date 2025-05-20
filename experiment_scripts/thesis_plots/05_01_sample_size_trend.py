import pandas as pd
import numpy as np

# Load CSV file
uniform_path = "experiment_scripts/thesis_plots/data/all_results_uniform_sampling.csv"
DATASETS = [
    ("HO_Rome_Res8", "HO-Rome"),
    ("HO_Geolife_Res8", "HO-Geolife"),
    ("HO_NYC_Res9", "HO-NYC"),
]

df_all = pd.read_csv(uniform_path)

def harmonize(df):
    df["importance"] = df.get("importance", "").replace(
        {
            "coverage_diversity": "C.D.",
            "unified": "Uni.",
            "entropy": "Ent.",
        }
    )
    df["method"] = df.apply(
        lambda row: (
            f"TraceHiding ({row['importance']})"
            if row.get("method") == "TraceHiding"
            else row.get("method")
        ),
        axis=1,
    )
    return df

df_all = harmonize(df_all)

methods = [
    "TraceHiding (Ent.)",
    "NegGrad+",
    "Retraining",
]
display_methods = [
    "TraceHiding (Ent.)",
    "NegGrad+",
    "Retraining",
]

# Ensure correct indexing for methods
idx_th = display_methods.index("TraceHiding (Ent.)")
idx_ng = display_methods.index("NegGrad+")
idx_ret = display_methods.index("Retraining")


metrics = [
    ("UA", "UA_mean"),
    ("RA", "RA_mean"),
    ("TA", "TA_mean"),
    ("MIA", "MIA_delta_mean"),
]
sample_percents = [1, 5, 10, 20]

for ds_code, ds_pretty in DATASETS:
    df = df_all[
        (df_all["dataset"] == ds_code) & (df_all["method"].isin(methods))
    ]
    # Group and aggregate means
    agg = (
        df.groupby(["method", "sample_percent"])
        .agg({
            "UA_mean": "mean",
            "RA_mean": "mean",
            "TA_mean": "mean",
            "MIA_delta_mean": "mean",
            "Time": "mean",
        })
        .reset_index()
    )

    # Build table rows
    table_rows = {metric: [] for metric, _ in metrics}
    speedup_row = []

    for metric, col in metrics:
        for method_idx, method in enumerate(display_methods):
            row = []
            for perc in sample_percents:
                val = agg.loc[
                    (agg["method"] == method) & (agg["sample_percent"] == perc), col
                ]
                if len(val) == 1:
                    val_float = val.values[0] * 100
                    row.append(f"{val_float:.2f}")
                else:
                    row.append("--")
            table_rows[metric].append(row)

    for method_idx, method in enumerate(display_methods):
        row = []
        for perc in sample_percents:
            retrain_time = agg.loc[
                (agg["method"] == "Retraining") & (agg["sample_percent"] == perc), "Time"
            ]
            method_time = agg.loc[
                (agg["method"] == method) & (agg["sample_percent"] == perc), "Time"
            ]
            if len(retrain_time) == 1 and len(method_time) == 1 and method_time.values[0] > 0:
                speedup = retrain_time.values[0] / method_time.values[0]
                row.append(f"{speedup:.1f}")
            else:
                row.append("--")
        speedup_row.append(row)
    
    # Build LaTeX rows with bolding
    tex_rows = []
    for metric_name, _ in metrics:
        
        th_values_str = table_rows[metric_name][idx_th]
        ng_values_str = table_rows[metric_name][idx_ng]
        ret_values_str = table_rows[metric_name][idx_ret]

        formatted_th_row_parts = []
        formatted_ng_row_parts = []

        for i in range(len(sample_percents)):
            th_s = th_values_str[i]
            ng_s = ng_values_str[i]
            ret_s = ret_values_str[i]

            # Default to original string values
            th_display = th_s
            ng_display = ng_s

            if th_s != "--" and ng_s != "--" and ret_s != "--":
                try:
                    th_v = float(th_s)
                    ng_v = float(ng_s)
                    ret_v = float(ret_s)

                    diff_th = abs(th_v - ret_v)
                    diff_ng = abs(ng_v - ret_v)

                    # Lower difference is better (closer)
                    if diff_th < diff_ng:
                        th_display = f"\\textbf{{{th_s}}}"
                    elif diff_ng < diff_th:
                        ng_display = f"\\textbf{{{ng_s}}}"
                    # If diff_th == diff_ng, neither is bolded specifically
                except ValueError:
                    # This might happen if parsing to float fails for an unexpected reason
                    pass 
            
            formatted_th_row_parts.append(th_display)
            formatted_ng_row_parts.append(ng_display)
        
        tex_row = f"& {metric_name} & " + " & ".join(formatted_th_row_parts)
        tex_row += " & " # Separator between TraceHiding and NegGrad+ columns
        tex_row += " & ".join(formatted_ng_row_parts)  
        tex_row += " \\\\"
        tex_rows.append(tex_row)
        
    # Add speedup row (unchanged logic for bolding, but ensure correct indexing if needed)
    # tex_row = "& Speedup & " + " & ".join(speedup_row[idx_th]) # Speedup for TraceHiding
    # tex_row += " & "
    # tex_row += " & ".join(speedup_row[idx_ng]) # Speedup for NegGrad+
    # tex_row += " \\\\"
    # tex_rows.append(tex_row)
    # Note: The original script's speedup row construction was commented out, 
    # and if re-enabled, ensure it aligns with the table structure.
    # If the table structure implies speedup is listed similarly, you might need:
    # speedup_th_str = " & ".join(speedup_row[idx_th])
    # speedup_ng_str = " & ".join(speedup_row[idx_ng])
    # tex_row_speedup = f"& Speedup & {speedup_th_str} & {speedup_ng_str} \\\\"
    # tex_rows.append(tex_row_speedup)

    contents = "\n".join(tex_rows)
    
    output_path = f"experiment_scripts/thesis_plots/outputs/sample_size_{ds_pretty}_rows_bolded.tex"
    with open(output_path, "w") as f:
        f.write(contents)
    print(f"Wrote {output_path}")
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

# Load the uploaded CSV
csv_path = "experiment_scripts/thesis_plots/data/all_results_uniform_sampling.csv"
df = pd.read_csv(csv_path)

# Map importance columns to their respective labels
df["importance"] = df["importance"].replace({
    "coverage_diversity": "C.D.",
    "unified": "Uni.",
    "entropy": "Ent.",
})

# Concatentate the imortance column to the method name if (method name is TracedHiding)
df["method"] = df.apply(
    lambda row: f"TraceHiding ({row['importance']})"
    if row["method"] == "TraceHiding" else row["method"],
    axis=1
)

# Keep only the eight methods of interest
methods_filter = [
    "TraceHiding (C.D.)",
    "TraceHiding (Ent.)",
    "TraceHiding (Uni.)",
    "NegGrad+",
    "NegGrad",
    "SCRUB",
    "Bad-T",
    "Finetuning",
]

df = df[df["method"].isin(methods_filter)].copy()

# Ensure numeric columns
df["UA_mean"] = df["UA_mean"].astype(float)
df["RA_mean"] = df["RA_mean"].astype(float)
df["sample_percent"] = df["sample_percent"].astype(float)

# Compute ranks within each (dataset, model, sample_percent) block (higher UA == better)
def rank_block(block):
    block = block.copy()
    block["rank"] = block["UA_mean"].rank(method="average", ascending=False)
    return block

df = df.groupby(["dataset", "model", "sample_percent"]).apply(rank_block).reset_index(drop=True)

# Average ranks across the 12 blocks for each method & sample_percent
avg_rank = (
    df.groupby(["method", "sample_percent"], as_index=False)["rank"]
      .mean()
      .rename(columns={"rank": "avg_rank"})
)

# Pivot to wide format for LaTeX
rank_table = avg_rank.pivot(index="method", columns="sample_percent", values="avg_rank")
rank_table = rank_table[[1.0, 5.0, 10.0, 20.0]]  # ensure correct order
rank_table.columns = ["1 \\%", "5 \\%", "10 \\%", "20 \\%"]

# Order methods by overall mean rank
rank_table["overall"] = rank_table.mean(axis=1)
rank_table = rank_table.sort_values("overall").drop(columns="overall")

# Round to 2 decimals
rank_table = rank_table.round(2)

# round to 2 decimals
rank_table = rank_table.applymap(lambda x: f"{x:.2f}")
rank_table_tex = rank_table.to_latex(column_format="lcccc", escape=False, index=True)
tex_path = "experiment_scripts/thesis_plots/outputs/rank_table.tex"
with open(tex_path, "w") as f:
    f.write(rank_table_tex)

print(f"Rank table saved to {tex_path}")
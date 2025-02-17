import pandas as pd


# DATASET_NAME = "HO_Rome_Res8"
# DATASET_PATH = "/local/data1/shared_data/higher_order_trajectory/rome/ho_rome_res8.csv"
# SEQUENCE_COLUMN = "higher_order_trajectory"

# DATASET_NAME = "HO_Porto_Res8"
# DATASET_PATH = "/local/data1/shared_data/higher_order_trajectory/porto/ho_porto_res8.csv"
# SEQUENCE_COLUMN = "higher_order_trajectory"

# DATASET_NAME = "HO_Geolife_Res8"
# DATASET_PATH = "/local/data1/shared_data/higher_order_trajectory/geolife/ho_geolife_res8.csv"
# SEQUENCE_COLUMN = "higher_order_trajectory"


DATASET_NAME = "HO_NYC_Res9"
DATASET_PATH = "datasets/nyc_HO_full.csv"
SEQUENCE_COLUMN = "poi"




# Load the CSV file
df = pd.read_csv(DATASET_PATH)

sequences = df[SEQUENCE_COLUMN].dropna().tolist()

# Save as a text file (one sequence per line) for tokenizer training
with open(f"datasets/plain_sequence_corp_{DATASET_NAME}.txt", "w") as f:
    for seq in sequences:
        f.write(seq + "\n")
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import logging
from utility.CheckInDataset import HexagonCheckInUserDataset


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


DATASET_NAME = "HO_NYC_Checkins"
TRAJECTORY_COLUMN = 'poi'
USER_COLUMN_NAME = 'user'
SPLIT_RATIO = 0.8
DATASET_CSV_RAW_PATH = "datasets/nyc_HO_full.csv"
RANDOM_STATE = 77

os.makedirs(f"experiments/{DATASET_NAME}/splits", exist_ok=True)

dataset = HexagonCheckInUserDataset(
    DATASET_CSV_RAW_PATH,
    user_id_col_name=USER_COLUMN_NAME,
    trajectory_col_name=TRAJECTORY_COLUMN,
    split_ratio=SPLIT_RATIO,
    random_state=RANDOM_STATE)

stats = {
    "users_size": dataset.users_size,
    "vocab_size": dataset.vocab_size,
    "train_size": len(dataset.get_train_data()),
    "test_size": len(dataset.get_test_data())
}


logging.info(f"Number of users: {stats['users_size']}")
logging.info(f"Number of cells: {stats['vocab_size']}")
logging.info(f"Number of training samples: {stats['train_size']}")
logging.info(f"Number of testing samples: {stats['test_size']}")

torch.save(dataset.get_train_data(), f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt")
torch.save(dataset.get_test_data(), f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt")
torch.save(dataset.cell_to_id, f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_cell_to_id.pt")
json.dump(stats, open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "w"))

logging.info(f"Training data, Testing data, and their stats are saved to experiments/{DATASET_NAME}/splits/")

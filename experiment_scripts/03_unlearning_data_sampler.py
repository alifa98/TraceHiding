"""
    This script samples data from the training data to create the unlearning and remaining datasets.
    this scripts only saved the index of the samples not copyinig the data and saving it again, so the data is not duplicated
    We can load the data via Subset and passing the indices of the samples to be used for unlearning and remaining.
"""
import json
import logging
import os
import pandas as pd
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utility.functions import custom_collate_fn
from torch.utils.data import DataLoader
from utility.CoverageDiversityImportance import CoverageDiversityImportance
from utility.EntropyImportance import EntropyImportance
from utility.LengthImportance import TrajectoryLengthImportance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#
# HO_Rome_Res8-217:     2, 10, 21, 43; 
# HO_Geolife_Res8-26:   1, 2, 3, 5; 
# HO_NYC_Res9-233:      2, 11, 23, 46; 
# HO_Porto_Res8-438:    4, 21, 43, 88; 

SAMPLE_BASED_ON_IMPORTANCE = True
IMPORTANCE_NAME = "entropy" # entropy, coverage, length
AGGREGATION_FUNCTION = "sum" # sum, mean, max

DATASET_NAME = "HO_Rome_Res8"

RANDOM_SAMPLE_UNLEARNING_SIZES = [270] # 5% of the training data
USER_UNLEARNING_SAMPLE_SIZE = [1, 5, 10, 20] # 1, 5, 10, 20 users (up to 10% of the users)

REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

os.makedirs(f"experiments/{DATASET_NAME}/unlearning/", exist_ok=True)
stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))

if SAMPLE_BASED_ON_IMPORTANCE:
    train_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
    test_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)

    entropy_calculator = EntropyImportance()
    coverage_calculator = CoverageDiversityImportance()
    length_calculator = TrajectoryLengthImportance()

    entropy_calculator.prepare(train_data + test_data)
    coverage_calculator.prepare(train_data + test_data)
    length_calculator.prepare(train_data + test_data)

    train_dloader = DataLoader(train_data, batch_size=len(train_data), collate_fn=custom_collate_fn, num_workers=24)
    test_dloader = DataLoader(test_data, batch_size=len(test_data), collate_fn=custom_collate_fn, num_workers=24)

    all_entropy_importances = []
    all_coverage_importances = []
    all_length_importances = []
    all_users = []

    for batch in train_dloader:
        
        batch_importance = entropy_calculator.calculate_importance(batch)
        all_entropy_importances.extend(batch_importance)
        
        batch_importance = coverage_calculator.calculate_importance(batch)
        all_coverage_importances.extend(batch_importance)
        
        batch_importance = length_calculator.calculate_importance(batch)
        all_length_importances.extend(batch_importance)
        
        all_users.extend(batch[1].tolist())


    data = pd.DataFrame({
        'entropy': all_entropy_importances,
        'coverage': all_coverage_importances,
        'length': all_length_importances,
        'user': all_users
        })
    
    # we should group by user so we can sample the users based on the importance
    if AGGREGATION_FUNCTION == "sum":
        user_importance = data.groupby('user').sum()
    elif AGGREGATION_FUNCTION == "mean":
        user_importance = data.groupby('user').mean()
    elif AGGREGATION_FUNCTION == "max":
        user_importance = data.groupby('user').max()

# the user_ids are from 0 to number_of_users (see HexagonCheckInUserDataset)
for sample_size in USER_UNLEARNING_SAMPLE_SIZE:
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
        
        base_dir = f"experiments/{DATASET_NAME}/unlearning/user_sample{f"_biased_{IMPORTANCE_NAME}_{AGGREGATION_FUNCTION}" if SAMPLE_BASED_ON_IMPORTANCE else ""}/sample_size_{sample_size}/sample_{i}/data/"
        os.makedirs(base_dir, exist_ok=True)
        
        if SAMPLE_BASED_ON_IMPORTANCE:
            random_user_ids = user_importance.sample(n=sample_size, weights=IMPORTANCE_NAME, axis=0).index.tolist()
        else:
            random_user_ids = torch.randperm(stats['users_size']).tolist()
            random_user_ids = random_user_ids[:sample_size]

        # data format is a list of tuples (sequence_list, user_id)
        training_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
       
        unlearning_indexes = [i for i, (_, u_id) in enumerate(training_dataset) if u_id in random_user_ids]
        remaining_indexes = [i for i, (_, u_id) in enumerate(training_dataset) if u_id not in random_user_ids]
        torch.save(unlearning_indexes, base_dir + "unlearning.indexes.pt")
        torch.save(remaining_indexes, base_dir + "remaining.indexes.pt")
        logging.info(f"User sampling of size {sample_size} done for sample {i}" + (f" biased by {IMPORTANCE_NAME}_{AGGREGATION_FUNCTION}" if SAMPLE_BASED_ON_IMPORTANCE else ""))


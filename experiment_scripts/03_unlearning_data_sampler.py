"""
    This script samples data from the training data to create the unlearning and remaining datasets.
    this scripts only saved the index of the samples not copyinig the data and saving it again, so the data is not duplicated
    We can load the data via Subset and passing the indices of the samples to be used for unlearning and remaining.
"""
import json
import logging
import os
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

DATASET_NAME = "Ho_Foursquare_NYC"
RANDOM_SAMPLE_UNLEARNING_SIZES = [200] # 5% of the training data
USER_UNLEARNING_SAMPLE_SIZE = [1, 5, 10, 20] # 1, 5, 10, 20 users (up to 10% of the users)

REPETITIONS_OF_EACH_SAMPLE_SIZE = 5


# ------------------------------------- START CONFIGURATIONS -------------------------------------#

os.makedirs(f"experiments/{DATASET_NAME}/unlearning/", exist_ok=True)
stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))


# SAMPLE RANDOM INDEXES FOR UNLEARNING
for sample_size in RANDOM_SAMPLE_UNLEARNING_SIZES:
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
        os.makedirs(f"experiments/{DATASET_NAME}/unlearning/random_sample/sample_size_{sample_size}/sample_{i}/data/", exist_ok=True)

for sample_size in RANDOM_SAMPLE_UNLEARNING_SIZES:
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
        random_permutaion = torch.randperm(stats['train_size']).tolist()
        unlearning_indexes = random_permutaion[:sample_size]
        remaining_indexes = random_permutaion[sample_size:]
        torch.save(unlearning_indexes, f"experiments/{DATASET_NAME}/unlearning/random_sample/sample_size_{sample_size}/sample_{i}/data/unlearning.indexes.pt")
        torch.save(remaining_indexes, f"experiments/{DATASET_NAME}/unlearning/random_sample/sample_size_{sample_size}/sample_{i}/data/remaining.indexes.pt")
        logging.info(f"Random sampling of size {sample_size} done for sample {i}")

# SAMPLE USER INDEXES FOR UNLEARNING
for sample_size in USER_UNLEARNING_SAMPLE_SIZE:
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
        os.makedirs(f"experiments/{DATASET_NAME}/unlearning/user_sample/sample_size_{sample_size}/sample_{i}/data/", exist_ok=True)


# the user_ids are from 0 to number_of_users (see HexagonCheckInUserDataset)
for sample_size in USER_UNLEARNING_SAMPLE_SIZE:
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
        random_user_ids = torch.randperm(stats['users_size']).tolist()
        random_user_ids = random_user_ids[:sample_size]

        # data format is a list of tuples (sequence_list, user_id)
        training_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
       
        unlearning_indexes = [i for i, (_, u_id) in enumerate(training_dataset) if u_id in random_user_ids]
        remaining_indexes = [i for i, (_, u_id) in enumerate(training_dataset) if u_id not in random_user_ids]
        torch.save(unlearning_indexes, f"experiments/{DATASET_NAME}/unlearning/user_sample/sample_size_{sample_size}/sample_{i}/data/unlearning.indexes.pt")
        torch.save(remaining_indexes, f"experiments/{DATASET_NAME}/unlearning/user_sample/sample_size_{sample_size}/sample_{i}/data/remaining.indexes.pt")
        logging.info(f"User sampling of size {sample_size} done for sample {i}")

# This script samples data from the training data to create the unlearning and remaining datasets.
# this scripts only saved the index of the samples not copyinig the data and saving it again, so the data is not duplicated
# We can load the data via Subset and passing the indices of the samples to be used for unlearning and remaining.
import json
import logging
import os
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


DATASET_NAME = "nyc_checkins"
RANDOM_SAMPLE_UNLEARNING_SIZES = [10, 20, 50, 100, 200, 300, 600, 1000]
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

NUMBER_OF_USER_UNLEARNING_SAMPLES = 5

os.makedirs(f"experiments/{DATASET_NAME}/unlearning/", exist_ok=True)

for sample_size in RANDOM_SAMPLE_UNLEARNING_SIZES:
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
        os.makedirs(
            f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/", exist_ok=True)


stats = json.load(
    open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))

# the user_ids are from 0 to number_of_users (see HexagonCheckInUserDataset)

for sample_size in RANDOM_SAMPLE_UNLEARNING_SIZES:
    logging.info(f"Sampling {sample_size} random indexes")
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
        random_permutaion = torch.randperm(stats['train_size']).tolist()
        unlearning_indexes = random_permutaion[:sample_size]
        remaining_indexes = random_permutaion[sample_size:]
        torch.save(unlearning_indexes,
                   f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/unlearning.indexes.pt")
        torch.save(remaining_indexes,
                   f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/remaining.indexes.pt")

        logging.info(f"Sample {i} done")


# sample user ids for unlearning
random_user_ids = torch.randperm(stats['users_size']).tolist()
random_user_ids = random_user_ids[:NUMBER_OF_USER_UNLEARNING_SAMPLES]

training_dataset = torch.load(
    f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt")

# data format is a list of tuples (sequence_list, user_id)
for user_id in random_user_ids:
    logging.info(f"Sampling user {user_id}")
    os.makedirs(
        f'experiments/{DATASET_NAME}/unlearning/user_sample/user_{user_id}/data', exist_ok=True)
    unlearning_indexes = [i for i, (_, u_id) in enumerate(
        training_dataset) if u_id == user_id]
    remaining_indexes = [i for i, (_, u_id) in enumerate(
        training_dataset) if u_id != user_id]
    torch.save(unlearning_indexes,
               f"experiments/{DATASET_NAME}/unlearning/user_sample/user_{user_id}/data/unlearning.indexes.pt")
    torch.save(remaining_indexes,
               f"experiments/{DATASET_NAME}/unlearning/user_sample/user_{user_id}/data/remaining.indexes.pt")

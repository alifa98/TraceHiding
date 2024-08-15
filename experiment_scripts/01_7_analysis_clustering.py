import os
import matplotlib.pyplot as plt
import torch

DATASET_NAME = "HO_Rome_Res8"

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)

os.makedirs(f"analysis/{DATASET_NAME}/plots", exist_ok=True)

def analyze_and_visualize(dataset, dataset_name="Dataset"):
   pass 


analyze_and_visualize(train_dataset, "Train Dataset")
analyze_and_visualize(test_dataset, "Test Dataset")
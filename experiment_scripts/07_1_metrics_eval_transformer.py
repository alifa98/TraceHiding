import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utility.evaluationUtils import get_model_outputs
from utility.functions import custom_collate_fn
from utility.ArguemntParser import get_args
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import logging
import torch
from utility.Dataset import CustomDataset
from utility.ArguemntParser import get_args
from utility.evaluationUtils import compute_metrics, get_model_outputs
from utility.functions import check_stopping_criteria, custom_collator_transformer
from transformers import ModernBertForSequenceClassification, BertForSequenceClassification
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import concurrent.futures
import logging
import torch
import json
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall, F1Score

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Helps with debugging CUDA errors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

args = get_args()
MODEL_NAME = args.model
DATASET_NAME = args.dataset
BASELINE_METHOD = args.method
IMPORTANCE_NAME = args.importance
SCENARIO = args.scenario
SAMPLE_SIZE =args.sampleSize
BATCH_SIZE = args.batchSize
EPOCH_NUM_TO_EVALUATE = args.epochIndex
BIASED_SAMPLE_IMPORTANCE_NAME = args.biased # if it is None, then the sample is not biased

REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

# ---------------------------------------------------
dynamic_epoch_index= EPOCH_NUM_TO_EVALUATE

# Load the dataset
train_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)
dataset_stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))

def tokenize_function(item):
        return {"input_ids": item[0], "attention_mask": [1] * len(item[0]), "labels": item[1]}
    
with concurrent.futures.ProcessPoolExecutor() as executor:
    logging.info("Tokenizing Training Data")
    train_data = list(executor.map(tokenize_function, train_data))
    logging.info("Tokenizing Test Data")
    test_data = list(executor.map(tokenize_function, test_data), total=len(test_data), desc="Tokenizing test data"))

for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
    
    # base folder
    base_folder = f"experiments/{DATASET_NAME}/unlearning/{SCENARIO}_sample{f"_biased_{BIASED_SAMPLE_IMPORTANCE_NAME}" if BIASED_SAMPLE_IMPORTANCE_NAME else ""}/sample_size_{SAMPLE_SIZE}/sample_{i}"

    unlearning_indices = torch.load(f"{base_folder}/data/unlearning.indexes.pt", weights_only=False)
    remaining_indices = torch.load(f"{base_folder}/data/remaining.indexes.pt", weights_only=False)

    unlearning_dataset = Subset(train_data, unlearning_indices)
    remaining_dataset = Subset(train_data, remaining_indices)

    unlearning_dataset = CustomDataset(unlearning_dataset)
    remaining_dataset = CustomDataset(remaining_dataset)
    test_dataset = CustomDataset(test_data)
    
    unlearning_dloader = DataLoader(unlearning_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collator_transformer, shuffle=True, num_workers=48)
    remaining_dloader = DataLoader(remaining_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collator_transformer, shuffle=True, num_workers=48)
    test_dloader = DataLoader(test_dataset, batch_size=len(test_data), collate_fn=custom_collator_transformer, num_workers=48)

    
    if BASELINE_METHOD == "original":
        baseline_model_path = f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt"
        results_folder = f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/evaluation"
    elif BASELINE_METHOD == "retraining":
        baseline_model_path = f"{base_folder}/{MODEL_NAME}/retraining/retrained_{MODEL_NAME}_model.pt"
        results_folder = f"{base_folder}/{MODEL_NAME}/{BASELINE_METHOD}/evaluation"
    elif BASELINE_METHOD == "trace_hiding":
        if not EPOCH_NUM_TO_EVALUATE:
            unlearning_stats = json.load(open(f"{base_folder}/{MODEL_NAME}/{BASELINE_METHOD}/{IMPORTANCE_NAME}/unlearning_stats-batch_size_{BATCH_SIZE}.json", "r"))
            max_epoch = max(unlearning_stats.keys())
            dynamic_epoch_index = int(max_epoch)
        baseline_model_path = f"{base_folder}/{MODEL_NAME}/{BASELINE_METHOD}/{IMPORTANCE_NAME}/unlearned_{MODEL_NAME}_epoch_{dynamic_epoch_index}_batch_{BATCH_SIZE}.pt"
        results_folder = f"{base_folder}/{MODEL_NAME}/{BASELINE_METHOD}/{IMPORTANCE_NAME}/evaluation"
    else:
        if not EPOCH_NUM_TO_EVALUATE:
            unlearning_stats = json.load(open(f"{base_folder}/{MODEL_NAME}/{BASELINE_METHOD}/unlearning_stats-batch_size_{BATCH_SIZE}.json", "r"))
            max_epoch = max(unlearning_stats.keys())
            dynamic_epoch_index = int(max_epoch)
        baseline_model_path = f"{base_folder}/{MODEL_NAME}/{BASELINE_METHOD}/unlearned_{MODEL_NAME}_epoch_{dynamic_epoch_index}_batch_{BATCH_SIZE}.pt"
        results_folder = f"{base_folder}/{MODEL_NAME}/{BASELINE_METHOD}/evaluation"
    
    os.makedirs(results_folder, exist_ok=True)
    
    if MODEL_NAME == "ModernBERT":
        baseline_model = ModernBertForSequenceClassification.from_pretrained(baseline_model_path)
    elif MODEL_NAME == "BERT":
        baseline_model = BertForSequenceClassification.from_pretrained(baseline_model_path)
    else:
        raise ValueError("Model name is not correct")
    
    baseline_model.eval()
    baseline_model.to(device)
    
    output_unlearning, unlearnin_true_labels = get_model_outputs(baseline_model, unlearning_dloader, device)
    output_remaining, remaining_true_labels = get_model_outputs(baseline_model, remaining_dloader, device)
    output_test, test_true_labels = get_model_outputs(baseline_model, test_dloader, device)
    
        
    # Assuming output_unlearning and output_test are logits
    accuracy_at_1 = Accuracy(task='multiclass', num_classes=int(dataset_stats['users_size']), top_k=1).to(device)
    accuracy_at_3 = Accuracy(task='multiclass', num_classes=int(dataset_stats['users_size']), top_k=3).to(device)
    accuracy_at_5 = Accuracy(task='multiclass', num_classes=int(dataset_stats['users_size']), top_k=5).to(device)
    precision = Precision(task='multiclass', num_classes=int(dataset_stats['users_size']), average='weighted').to(device)
    recall = Recall(task='multiclass', num_classes=int(dataset_stats['users_size']), average='weighted').to(device)
    f1_score = F1Score(task='multiclass', num_classes=int(dataset_stats['users_size']), average='weighted').to(device)
    
    metrics = {}
    metrics['unlearning_dataset'] = {
        'accuracy@1': accuracy_at_1(torch.argmax(output_unlearning, dim=-1), unlearnin_true_labels).item(),
        'accuracy@3': accuracy_at_3(output_unlearning, unlearnin_true_labels).item(),
        'accuracy@5': accuracy_at_5(output_unlearning, unlearnin_true_labels).item(),
        'precision': precision(torch.argmax(output_unlearning, dim=-1), unlearnin_true_labels).item(),
        'recall': recall(torch.argmax(output_unlearning, dim=-1), unlearnin_true_labels).item(),
        'f1_score': f1_score(torch.argmax(output_unlearning, dim=-1), unlearnin_true_labels).item()
    }
    
    # reset the metrics
    accuracy_at_1.reset()
    accuracy_at_3.reset()
    accuracy_at_5.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()
    
    metrics['test_dataset'] = {
        'accuracy@1': accuracy_at_1(torch.argmax(output_test, dim=-1), test_true_labels).item(),
        'accuracy@3': accuracy_at_3(output_test, test_true_labels).item(),
        'accuracy@5': accuracy_at_5(output_test, test_true_labels).item(),
        'precision': precision(torch.argmax(output_test, dim=-1), test_true_labels).item(),
        'recall': recall(torch.argmax(output_test, dim=-1), test_true_labels).item(),
        'f1_score': f1_score(torch.argmax(output_test, dim=-1), test_true_labels).item()
    }
    
    # reset the metrics
    accuracy_at_1.reset()
    accuracy_at_3.reset()
    accuracy_at_5.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()
    
    metrics['remaining_dataset'] = {
        'accuracy@1': accuracy_at_1(torch.argmax(output_remaining, dim=-1), remaining_true_labels).item(),
        'accuracy@3': accuracy_at_3(output_remaining, remaining_true_labels).item(),
        'accuracy@5': accuracy_at_5(output_remaining, remaining_true_labels).item(),
        'precision': precision(torch.argmax(output_remaining, dim=-1), remaining_true_labels).item(),
        'recall': recall(torch.argmax(output_remaining, dim=-1), remaining_true_labels).item(),
        'f1_score': f1_score(torch.argmax(output_remaining, dim=-1), remaining_true_labels).item()
    }
    
    json.dump(metrics, open(f"{results_folder}/metrics_performance_epoch_{EPOCH_NUM_TO_EVALUATE}.json", "w"), indent=4)
    
    print(f"Metrics for sample_{i} are saved in {results_folder}/metrics_performance_epoch_{EPOCH_NUM_TO_EVALUATE}.json")
    
    if BASELINE_METHOD == 'original':
        # Do not repeat for the REPETITIONS_OF_EACH_SAMPLE_SIZE times.
        break

import json
import os
import sys
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utility.evaluationUtils import get_model_outputs
from utility.ArguemntParser import get_args
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from utility.Dataset import CustomDataset
from utility.ArguemntParser import get_args
from utility.evaluationUtils import get_model_outputs
from utility.functions import custom_collator_transformer
from transformers import ModernBertForSequenceClassification, BertForSequenceClassification
from torch.utils.data import DataLoader
import concurrent.futures
import logging
import torch
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
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
    test_data = list(executor.map(tokenize_function, test_data))

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
    unlearning_dloader = DataLoader(unlearning_dataset, batch_size=500, collate_fn=custom_collator_transformer, shuffle=True, num_workers=48)
    remaining_dloader = DataLoader(remaining_dataset, batch_size=500, collate_fn=custom_collator_transformer, shuffle=True, num_workers=48)
    test_dloader = DataLoader(test_dataset, batch_size=500, collate_fn=custom_collator_transformer, num_workers=48)
    
    logging.info(f"Data Loaders are ready for sample {i}")
    
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
    
    logging.info(f"Model is loaded from: {baseline_model_path}")
    
    baseline_model.eval()
    baseline_model.to(device)
    
    logits_unlearning, _ = get_model_outputs(baseline_model, unlearning_dloader, device)
    # logits_test, _ = get_model_outputs(baseline_model, test_dloader, device)
    logits_remaining, _ = get_model_outputs(baseline_model, remaining_dloader, device)
    
    labels_unlearning = torch.zeros(logits_unlearning.shape[0]).to(device)
    # labels_test = torch.zeros(logits_test.shape[0]).to(device)
    labels_remaining = torch.ones(logits_remaining.shape[0]).to(device)
    
    X = np.vstack([logits_remaining.cpu(), logits_unlearning.cpu()])
    y = np.concatenate([labels_remaining.cpu(), labels_unlearning.cpu()])
    
    
    # Split the data first to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Scale the training data and then transform the test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logestic Regression
    logging.info(f"Training Logistic Regression model on the training set...")
    model = LogisticRegression(random_state=42, max_iter=5000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    logging.info(f"Logistic Regression model trained.")
    y_train_pred_probs = model.predict_proba(X_train_scaled)[:, 1]
    
    # use candidate thresholds to find the best threshold in a binrary classification and f1 score
    candidate_thresholds = np.arange(0.0, 1.0, 0.01)
    f1_scores = []
    for threshold in candidate_thresholds:
        y_train_pred_labels = np.where(y_train_pred_probs > threshold, 1, 0)
        f1 = f1_score(y_train, y_train_pred_labels)
        f1_scores.append(f1)
    
    best_threshold = candidate_thresholds[np.argmax(f1_scores)]
    logging.info(f"Best threshold: {best_threshold}")
    
    # Get the predictions for the test set and evaluate the model
    y_pred_probs = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_labels = np.where(y_pred_probs > best_threshold, 1, 0)
    
    # Calculate Metrics
    auc_roc = roc_auc_score(y_test, y_pred_probs)
    accuracy = accuracy_score(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels)
    recall = recall_score(y_test, y_pred_labels)
    f1 = f1_score(y_test, y_pred_labels)

    save_dict = {
        'auc_roc': auc_roc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    json.dump(save_dict, open(f"{results_folder}/metrics_lr_mia_epoch_{EPOCH_NUM_TO_EVALUATE}.json", "w"), indent=4)

    logging.info(f"Membership Inference Attack metrics are saved in: {results_folder}/metrics_lr_mia_epoch_{EPOCH_NUM_TO_EVALUATE}.json")
    
    if BASELINE_METHOD == 'original':
        # Do not repeat for the REPETITIONS_OF_EACH_SAMPLE_SIZE times.
        break
    

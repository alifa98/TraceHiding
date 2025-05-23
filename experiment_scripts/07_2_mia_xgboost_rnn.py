import json
import os
import sys
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utility.ArguemntParser import get_args
from utility.evaluationUtils import get_model_outputs, js_divergence
from utility.functions import custom_collate_fn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import logging
import torch
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

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
train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)
dataset_stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))


for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
    
    # base folder
    base_folder = f"experiments/{DATASET_NAME}/unlearning/{SCENARIO}_sample{f"_biased_{BIASED_SAMPLE_IMPORTANCE_NAME}" if BIASED_SAMPLE_IMPORTANCE_NAME else ""}/sample_size_{SAMPLE_SIZE}/sample_{i}"

    unlearning_indices = torch.load(f"{base_folder}/data/unlearning.indexes.pt", weights_only=False)
    remaining_indices = torch.load(f"{base_folder}/data/remaining.indexes.pt", weights_only=False)

    unlearning_dataset = Subset(train_dataset, unlearning_indices)
    remaining_dataset = Subset(train_dataset, remaining_indices)
    unlearning_dloader = DataLoader(unlearning_dataset, batch_size=len(unlearning_dataset), collate_fn=custom_collate_fn, num_workers=24)
    remaining_dloader = DataLoader(remaining_dataset, batch_size=len(remaining_dataset), collate_fn=custom_collate_fn, num_workers=24)
    test_dloader = DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=custom_collate_fn, num_workers=24)
    
        
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
    
    baseline_model = torch.load(baseline_model_path, weights_only=False, map_location=device)
    baseline_model.eval()
    
    logits_unlearning, _ = get_model_outputs(baseline_model, unlearning_dloader, device)
    logits_test, _ = get_model_outputs(baseline_model, test_dloader, device)
    logits_remaining, _ = get_model_outputs(baseline_model, remaining_dloader, device)
    
    labels_unlearning = torch.zeros(logits_unlearning.shape[0]).to(device)
    labels_test = torch.zeros(logits_test.shape[0]).to(device)
    labels_remaining = torch.ones(logits_remaining.shape[0]).to(device)
    
    X = np.vstack([logits_remaining.cpu(), logits_unlearning.cpu(), logits_test.cpu()])
    y = np.concatenate([labels_remaining.cpu(), labels_unlearning.cpu(), labels_test.cpu()])
    
    
    # Split the data first to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Scale the training data and then transform the test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Undersample the training data
    undersampler = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = undersampler.fit_resample(X_train_scaled, y_train)

    # Train an XGBoost model for membership inference attack
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, device='cuda')
    logging.info("Training XGBoost model for Membership Inference Attack")
    xgb_model.fit(X_train_res, y_train_res)
    logging.info("Training is done")
    
    
    # Predict membership for validation set
    y_pred_probs = xgb_model.predict_proba(X_test)[:, 1]  # Probability of being a member (label=1)
    y_pred_labels = (y_pred_probs >= 0.5).astype(int)    # Thresholding at 0.5
    
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
    
    json.dump(save_dict, open(f"{results_folder}/metrics_xgboost_mia_epoch_{EPOCH_NUM_TO_EVALUATE}.json", "w"), indent=4)

    # conf_matrix = confusion_matrix(y_val, y_pred_labels)
    # # Plot confusion matrix
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=['Non-member', 'Member'], yticklabels=['Non-member', 'Member'], cbar=False)
    # plt.title('Confusion Matrix')
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')
    # plt.savefig(f"{results_folder}/confusion_matrix_epoch_{EPOCH_NUM_TO_EVALUATE}.pdf", bbox_inches='tight', format='pdf')
    # print(f"Confusion Matrix saved at: {results_folder}/confusion_matrix_epoch_{EPOCH_NUM_TO_EVALUATE}.pdf")

    # # ROC Curve
    # fpr, tpr, _ = roc_curve(y_val, y_pred_probs)
    # roc_auc = auc(fpr, tpr)

    # # Plot ROC Curve
    # plt.figure(figsize=(6, 4))
    # plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.4f})')
    # plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Diagonal line for random guessing
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc="lower right")
    # plt.savefig(f"{results_folder}/roc_curve_epoch_{EPOCH_NUM_TO_EVALUATE}.pdf", bbox_inches='tight', format='pdf')
    # print(f"ROC Curve saved at: {results_folder}/roc_curve_epoch_{EPOCH_NUM_TO_EVALUATE}.pdf")
    
    logging.info(f"Membership Inference Attack metrics are saved in: {results_folder}/metrics_xgboost_mia_epoch_{EPOCH_NUM_TO_EVALUATE}.json")
    
    if BASELINE_METHOD == 'original':
        # Do not repeat for the REPETITIONS_OF_EACH_SAMPLE_SIZE times.
        break

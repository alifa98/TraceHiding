import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utility.evaluationUtils import evaluate_mia_model, get_model_outputs, js_divergence, train_mia_model
from utility.functions import custom_collate_fn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import logging
import torch
from scipy.spatial.distance import euclidean
import numpy as np
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.base import accuracy_score


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Helps with debugging CUDA errors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


DATASET_NAME = "HO_Rome_Res8"
MODEL_NAME = "BERT"
BASELINE_METHOD = "original" # original, retrained, finetune, negrad, negradplus, badt, scrub

RANDOM_SAMPLE_UNLEARNING_SIZES = [600]
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

METRIC_NAMES = ["performance"] # ["performance", "activation_distance", "JS_divergence", "MIA"]
MIA_TYPE = "RF" # ["RF", "NN"]
# ---------------------------------------------------


# Load the dataset
train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt")
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt")
dataset_stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))


for sample_size in RANDOM_SAMPLE_UNLEARNING_SIZES:
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
        unlearning_indexes = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/unlearning.indexes.pt")
        remaining_indexes = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/remaining.indexes.pt")
        
        unlearning_dataset = Subset(train_dataset, unlearning_indexes)
        remaining_dataset = Subset(train_dataset, remaining_indexes)
        unlearning_dloader = DataLoader(unlearning_dataset, batch_size=len(unlearning_dataset), collate_fn=custom_collate_fn, num_workers=24)
        remaining_dloader = DataLoader(remaining_dataset, batch_size=len(remaining_dataset), collate_fn=custom_collate_fn, num_workers=24)
        test_dloader = DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=custom_collate_fn, num_workers=24)
        
        if BASELINE_METHOD == "original":
            baseline_model = torch.load(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt").to(device)
        elif BASELINE_METHOD == "retrained":
            baseline_model = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/retrained_{MODEL_NAME}_model.pt").to(device)
        else:
            # path to the trained baseline model
            # baseline_model_path = f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/baseline_{MODEL_NAME}_model.pt"
            raise NotImplementedError("Baseline method not implemented yet")
        
        baseline_model.eval()
        
        output_unlearning, unlearnin_true_labels = get_model_outputs(baseline_model, unlearning_dloader, device)
        output_test, test_true_labels = get_model_outputs(baseline_model, test_dloader, device)
        
        if "performance" in METRIC_NAMES:
            
            # Assuming output_unlearning and output_test are logits or probabilities
            accuracy_at_1 = Accuracy(task='multiclass', num_classes=int(dataset_stats['users_size']), top_k=1).to(device)
            accuracy_at_3 = Accuracy(task='multiclass', num_classes=int(dataset_stats['users_size']), top_k=3).to(device)
            accuracy_at_5 = Accuracy(task='multiclass', num_classes=int(dataset_stats['users_size']), top_k=5).to(device)
            precision = Precision(task='multiclass', num_classes=int(dataset_stats['users_size']), average='weighted').to(device)
            recall = Recall(task='multiclass', num_classes=int(dataset_stats['users_size']), average='weighted').to(device)
            f1_score = F1Score(task='multiclass', num_classes=int(dataset_stats['users_size']), average='weighted').to(device)
            
            metrics = {}
            metrics['performance for Unlearning Dataset'] = {
                'accuracy@1': accuracy_at_1(output_unlearning, unlearnin_true_labels).item(),
                'accuracy@3': accuracy_at_3(output_unlearning, unlearnin_true_labels).item(),
                'accuracy@5': accuracy_at_5(output_unlearning, unlearnin_true_labels).item(),
                'precision': precision(output_unlearning.argmax(dim=1), unlearnin_true_labels).item(),
                'recall': recall(output_unlearning.argmax(dim=1), unlearnin_true_labels).item(),
                'f1_score': f1_score(output_unlearning.argmax(dim=1), unlearnin_true_labels).item()
            }
            
            # reset the metrics
            accuracy_at_1.reset()
            accuracy_at_3.reset()
            accuracy_at_5.reset()
            precision.reset()
            recall.reset()
            f1_score.reset()
            
            metrics['performance for Test Dataset'] = {
                'accuracy@1': accuracy_at_1(output_test, test_true_labels).item(),
                'accuracy@3': accuracy_at_3(output_test, test_true_labels).item(),
                'accuracy@5': accuracy_at_5(output_test, test_true_labels).item(),
                'precision': precision(output_test.argmax(dim=1), test_true_labels).item(),
                'recall': recall(output_test.argmax(dim=1), test_true_labels).item(),
                'f1_score': f1_score(output_test.argmax(dim=1), test_true_labels).item()
            }

            print(f"Performance metrics for sample size {sample_size}, repetition {i}: {metrics}")
        
        # load the retrained model for distance & divergence calculation
        if "activation_distance" in METRIC_NAMES or "JS_divergence" in METRIC_NAMES:
            retrained_model = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/retrained_{MODEL_NAME}_model.pt").to(device)
            retrained_model.eval()
            retrained_output_unlearning, _ = get_model_outputs(retrained_model, unlearning_dloader, device)
          
        if "activation_distance" in METRIC_NAMES:
            activation_distances = [euclidean(r.cpu(), b.cpu()) for r, b in zip(retrained_output_unlearning, output_unlearning)]
            avg_activation_distance = np.mean(activation_distances)
            print(f"Average activation distance for sample size {sample_size}, repetition {i}: {avg_activation_distance}")
        
        if "JS_divergence" in METRIC_NAMES:
            js_divergences = [js_divergence(r.cpu().numpy(), b.cpu().numpy()) for r, b in zip(torch.softmax(retrained_output_unlearning, dim=1), torch.softmax(output_unlearning, dim=1))]
            avg_js_divergence = np.mean(js_divergences)
            print(f"Average JS divergence for sample size {sample_size}, repetition {i}: {avg_js_divergence}")
        
        if "MIA" in METRIC_NAMES:
            """
            The MIA model is trained on the remaining data and the test data.
            The final model should be able to infer if the unlearning data is part of the training data or not.
            the more accurate the model, the more the unlearning data is similar to the training data.
            The lables for the training data are 0 and for the test/unlearning data are 1.
            """
            
            output_remaining, _ = get_model_outputs(baseline_model, remaining_dloader, device)
            
            mia_train_data = torch.cat((output_remaining, output_test), dim=0).to(device)
            mia_train_labels = torch.cat((torch.zeros(len(output_remaining)), torch.ones(len(output_test))), dim=0).to(device)
            
            mia_test_data = output_unlearning.to(device)
            mia_test_labels = torch.ones(len(output_unlearning)).to(device) # we assume that the unlearning data should be classified as test (not training) data
            
            if MIA_TYPE == "NN":
                mia_train_data = torch.softmax(mia_train_data, dim=1)
                mia_test_data = torch.softmax(mia_test_data, dim=1)
                mia_train_dataset = torch.utils.data.TensorDataset(mia_train_data, mia_train_labels)
                mia_test_dataset = torch.utils.data.TensorDataset(mia_test_data, mia_test_labels)
            
                mia_train_dataloader = DataLoader(mia_train_dataset, batch_size=128, shuffle=True)
                mia_test_dataloader = DataLoader(mia_test_dataset, batch_size=len(mia_test_dataset))
                
                mia_model = train_mia_model(mia_train_dataloader, mia_test_data.shape[1], device)
                
                mia_accuracy = evaluate_mia_model(mia_model, mia_test_dataloader, device)
                
                print(f"MIA accuracy for sample size {sample_size}, repetition {i}: {mia_accuracy}")
            elif MIA_TYPE == "RF":
                from sklearn.ensemble import RandomForestClassifier
                from imblearn.over_sampling import SMOTE
                
                mia_train_data = mia_train_data.cpu().detach().numpy()
                mia_train_labels = mia_train_labels.cpu().detach().numpy().astype(np.int32) # Class labels 
                mia_test_data = mia_test_data.cpu().detach().numpy()
                mia_test_labels = mia_test_labels.cpu().detach().numpy().astype(np.int32) # Class labels
                
                smote = SMOTE()
                mia_train_data_resampled, mia_train_labels_resampled = smote.fit_resample(mia_train_data, mia_train_labels)

                logging.info("Random Forest Training...")
                mia_model = RandomForestClassifier()
                mia_model.fit(mia_train_data_resampled, mia_train_labels_resampled)
                
                mia_predictions = mia_model.predict(mia_test_data)
                mia_accuracy = accuracy_score(mia_test_labels, mia_predictions)
                
                print(f"MIA accuracy for sample size {sample_size}, repetition {i}: {mia_accuracy}")    
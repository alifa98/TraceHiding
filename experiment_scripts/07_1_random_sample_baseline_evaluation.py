import os
import sys

from sklearn.model_selection import train_test_split
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utility.evaluationUtils import evaluate_mia_model, get_model_outputs, js_divergence, train_mia_model
from utility.functions import custom_collate_fn
from torch.nn import functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import logging
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import euclidean
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Helps with debugging CUDA errors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


DATASET_NAME = "nyc_checkins"
MODEL_NAME = "LSTM"
BASELINE_METHOD = "retrained" # original, retrained, finetune, negrad, negradplus, badt, scrub

RANDOM_SAMPLE_UNLEARNING_SIZES = [600] #[10, 20, 50, 100, 200, 300, 600, 1000]
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

METRIC_NAMES = ["MIA"] # ["performance", "activation_distance", "JS_divergence", "MIA"]
MIA_TYPE = "RF" # ["RF", "NN"]
# ---------------------------------------------------


# Load the dataset
train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt")
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt")


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
        
        baseline_output_unlearning, _ = get_model_outputs(baseline_model, unlearning_dloader, device)
        
        if "performance" in METRIC_NAMES:
            baseline_predictions_unlearning = baseline_output_unlearning.argmax(dim=1)
            metrics = {
                'performance': {
                    'accuracy': accuracy_score(baseline_predictions_unlearning.cpu(), baseline_predictions_unlearning.cpu()),
                    'precision': precision_score(baseline_predictions_unlearning.cpu(), baseline_predictions_unlearning.cpu(), average='weighted'),
                    'recall': recall_score(baseline_predictions_unlearning.cpu(), baseline_predictions_unlearning.cpu(), average='weighted'),
                    'f1_score': f1_score(baseline_predictions_unlearning.cpu(), baseline_predictions_unlearning.cpu(), average='weighted')
                }
            }
            print(f"Performance metrics for sample size {sample_size}, repetition {i}: {metrics}")
        
        # load the retrained model for distance & divergence calculation
        if "activation_distance" in METRIC_NAMES or "JS_divergence" in METRIC_NAMES:
            retrained_model = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/retrained_{MODEL_NAME}_model.pt").to(device)
            retrained_model.eval()
            retrained_output_unlearning, _ = get_model_outputs(retrained_model, unlearning_dloader, device)
          
        if "activation_distance" in METRIC_NAMES:
            activation_distances = [euclidean(r.cpu(), b.cpu()) for r, b in zip(retrained_output_unlearning, baseline_output_unlearning)]
            avg_activation_distance = np.mean(activation_distances)
            print(f"Average activation distance for sample size {sample_size}, repetition {i}: {avg_activation_distance}")
        
        if "JS_divergence" in METRIC_NAMES:
            js_divergences = [js_divergence(r.cpu().numpy(), b.cpu().numpy()) for r, b in zip(torch.softmax(retrained_output_unlearning, dim=1), torch.softmax(baseline_output_unlearning, dim=1))]
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
            output_test, _ = get_model_outputs(baseline_model, test_dloader, device)
            
            mia_train_data = torch.cat((output_remaining, output_test), dim=0).to(device)
            mia_train_labels = torch.cat((torch.zeros(len(output_remaining)), torch.ones(len(output_test))), dim=0).to(device)
            
            mia_test_data = baseline_output_unlearning.to(device)
            mia_test_labels = torch.ones(len(baseline_output_unlearning)).to(device)
            
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
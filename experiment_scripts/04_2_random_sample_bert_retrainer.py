import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch.utils.data import Subset
import logging
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import BertConfig, BertForSequenceClassification
from torch.utils.data import Dataset
import json
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

MODEL_NAME = "BERT"
DATASET_NAME = "Ho_Foursquare_NYC"
# DATASET_NAME = "HO_Porto_Res8"
# DATASET_NAME = "HO_Geolife_Res8"
RANDOM_SAMPLE_UNLEARNING_SIZES =[200] # Rome:135, Porto: 45700, Geolife: 50
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

# ------------------------------------- END CONFIGURATIONS -------------------------------------#

# load model parameters
model_params = json.load(open(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.json", "r"))

HIDDEN_SIZE = model_params["hidden_size"]
NUM_HIDDEN_LAYERS = model_params["num_hidden_layers"]
NUM_ATTENTION_HEADS = model_params["num_attention_heads"]
INTERMEDIATE_SIZE = model_params["intermediate_size"]
MAXIMUM_POSITION_EMBEDDING = model_params["max_position_embeddings"]
BATCH_SIZE = model_params["batch_size"]
EPOCHS = 300

train_dataset_og = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset_og = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)
stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))

for sample_size in RANDOM_SAMPLE_UNLEARNING_SIZES:
    for i in range(1,REPETITIONS_OF_EACH_SAMPLE_SIZE):
        remaining_indexes = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/remaining.indexes.pt", weights_only=False)

        # LOAD DATASET
        reamining_dataset = Subset(train_dataset_og, remaining_indexes)

        # Check if a GPU is available and use it
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LOAD DATASET
        class HexagonDatasetForBert(Dataset):
            def __init__(self, sequences, lables):
                self.sequences = sequences
                self.labels = lables
                self.max_length = 0

            def __len__(self):
                return len(self.sequences)

            def __getitem__(self, idx):
                return {
                    'input_ids': torch.tensor(self.padded_sequences[idx], dtype=torch.long),
                    'attention_mask': torch.tensor([1 if token != 0 else 0 for token in self.padded_sequences[idx]], dtype=torch.long),
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                }

            def pad_a_sequence(self, sequence):
                if len(sequence) < self.max_length:
                    return sequence + [0] * (self.max_length - len(sequence))
                return sequence[:self.max_length]
            
            def pad_sequences(self):
                self.padded_sequences = [self.pad_a_sequence(seq) for seq in self.sequences]


        sequences, labels = zip(*reamining_dataset)
        train_dataset = HexagonDatasetForBert(sequences, labels)
        sequences, labels = zip(*test_dataset_og)
        test_dataset = HexagonDatasetForBert(sequences, labels)

        max_sequence_length = max(max(len(seq) for seq in test_dataset.sequences), max(len(seq) for seq in train_dataset.sequences))
        train_dataset.max_length = max_sequence_length
        test_dataset.max_length = max_sequence_length

        train_dataset.pad_sequences()
        test_dataset.pad_sequences()

        # model
        config = BertConfig(
            vocab_size=stats["vocab_size"]+2,
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            intermediate_size=INTERMEDIATE_SIZE,
            max_position_embeddings=MAXIMUM_POSITION_EMBEDDING,
            num_labels=int(stats["users_size"]),
        )

        model = BertForSequenceClassification(config)

        # Move the model to the specified device
        model.to(device)

        # results folder
        results_folder = f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/retraining"
        os.makedirs(results_folder, exist_ok=True)
        
        CHECKPOINT_DIR = f"{results_folder}/checkpoints/"
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=CHECKPOINT_DIR,
            run_name=f"{MODEL_NAME}_{DATASET_NAME}",
            eval_strategy="epoch",  # Use eval_strategy instead of evaluation_strategy
            save_strategy="epoch",  # Ensure save_strategy matches eval_strategy
            learning_rate=1e-5,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=0.02,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to=None,
        )
        
        # Define the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]
        )


        # train the model
        logging.info(f'Training the model for the remaining data, unlearning sample size: {sample_size}, repetition: {i}')
        trainer.train()
        
        # save the model
        torch.save(model, f"{results_folder}/retrained_{MODEL_NAME}_model.pt")
        logging.info(f'Model is saved for the remaining data, unlearning sample size: {sample_size}, sample: {i}')

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from models.HoLSTM import LitHigherOrderLSTM
from models.HoGRU import LitHigherOrderGRU
from utility.functions import custom_collate_fn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import logging
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import TensorBoardLogger

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "LSTM"
DATASET_NAME = "HO_Rome_Res8-v2"

# Hyperparameter Search Configuration
MAX_EPOCHS = 300

# Load datasets and stats
train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt")
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt")
cell_to_id = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_cell_to_id.pt")
stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))

train_dloader = DataLoader(train_dataset, batch_size=20, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
test_dloader = DataLoader(test_dataset, batch_size=20, collate_fn=custom_collate_fn, num_workers=24)

os.makedirs(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/checkpoints", exist_ok=True)
os.makedirs(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/optuna_logs", exist_ok=True)

# Define the objective function for Optuna
def objective(trial):

    EMBEDDING_SIZE = trial.suggest_int('embedding_size', 100, 500)
    HIDDEN_SIZE = trial.suggest_int('hidden_size', 50, 600)
    NUMBER_OF_LAYERS = trial.suggest_int('number_of_layers', 1, 3)
    DROPOUT = trial.suggest_uniform('dropout', 0.0, 0.5)
    BATCH_SIZE = trial.suggest_int('batch_size', 10, 300)

    # Recreate data loaders with new batch size
    train_dloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
    test_dloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, num_workers=24)

    if MODEL_NAME == "LSTM":
        model = LitHigherOrderLSTM(stats['vocab_size'], stats['users_size'], EMBEDDING_SIZE, HIDDEN_SIZE, NUMBER_OF_LAYERS, DROPOUT)
    elif MODEL_NAME == "GRU":
        model = LitHigherOrderGRU(stats['vocab_size'], stats['users_size'], EMBEDDING_SIZE, HIDDEN_SIZE, NUMBER_OF_LAYERS, DROPOUT)
    else:
        raise ValueError("Model name is not valid")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/checkpoints",
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        filename='{epoch}-{val_loss:.2f}',
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )

    # pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    # Initialize Trainer with the correct callbacks
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=MAX_EPOCHS,
        enable_progress_bar=True,
        callbacks=[early_stopping_callback, checkpoint_callback],  # Ensure proper handling of callbacks
        logger=TensorBoardLogger(save_dir=f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/optuna_logs")
    )

    trainer.fit(model, train_dloader, test_dloader)

    val_loss = trainer.callback_metrics["val_loss"].item()

    # Implement manual pruning logic
    trial.report(val_loss, step=trainer.current_epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_loss


# Run Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, n_jobs=10)

# Save the best model and hyperparameters
best_trial = study.best_trial
# torch.save(model, f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/best_model_{MODEL_NAME}.pt")
json.dump(best_trial.params, open(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/best_params.json", "w"))

# Optional: visualize the study
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)


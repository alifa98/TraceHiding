import json
import logging
from typing import Any, Mapping
from pytorch_lightning.callbacks.callback import Callback
from lightning import LightningModule, Trainer
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from torch import Tensor


class EachBatchTester(Callback):

    batch_number = 0

    def __init__(self, dataLoader, logging_path):
        super().__init__()
        self.dataloader = dataLoader
        self.logging_file = logging_path
        self.accuracies = {}
    
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info(f"Starting the logging of each mini-batch test results in the path: {self.logging_file}")
        return super().on_train_start(trainer, pl_module)
    
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int) -> None:
        self.batch_number += 1
        acc1, acc3, acc5, precision, recall, f1 = pl_module.test_model(self.dataloader)
        self.accuracies[self.batch_number] = {"accuracy1": acc1.item(), "accuracy3": acc3.item(), "accuracy5": acc5.item(), "precision": precision.item(), "recall": recall.item(), "f1": f1.item()}
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        
        #dump the accuracies to a json file:
        with open(self.logging_file, "w") as f:
            json.dump(self.accuracies, f)
        
        return super().on_train_end(trainer, pl_module)
        
        
import logging
import torch
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertForSequenceClassification, ModernBertForSequenceClassification
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

def get_class_weights(labels):
    class_counts = torch.bincount(labels.int())
    total_samples = len(labels)
    class_weights = total_samples / class_counts.float()
    return class_weights

def get_model_outputs(model, dataloader, device):
    model_outputs = []
    true_labels = []
    with torch.no_grad():
        for batch in dataloader:
            
            # get the model outputs
            if isinstance(model, BertForSequenceClassification) or isinstance(model, ModernBertForSequenceClassification):
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)
                outputs = model(**inputs, labels=labels)
            elif isinstance(model, torch.nn.Module):
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
            else:
                raise ValueError("Model not supported")
            
            #now get the logits and append them to the list
            if isinstance(outputs, torch.Tensor):
                model_outputs.append(outputs)
            elif isinstance(outputs, SequenceClassifierOutput):
                # for BERT models
                model_outputs.append(outputs.logits)
            
            true_labels.append(labels)
    return torch.cat(model_outputs), torch.cat(true_labels)

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def train_nn_mia_model(mia_data_loader, in_feature_size, device):
    
    class MIA(torch.nn.Module):
        def __init__(self):
            super(MIA, self).__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(in_feature_size, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 1),
            )
        
        def forward(self, x):
            return self.model(x)
    
    mia_model = MIA().to(device)
    class_weights = get_class_weights(mia_data_loader.dataset.tensors[1])
    loss_fn =  torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    optimizer = torch.optim.Adam(mia_model.parameters(), lr=0.001)
    
    # early stopping with patience of 5 epochs
    best_loss = float('inf')
    best_model = None
    no_improvement = 0
    for epoch in range(100):
        mia_model.train()
        pbar = tqdm(mia_data_loader, desc=f"Training MIA model, epoch {epoch + 1}", leave=False)
        for batch in pbar:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = mia_model(inputs)
            loss = loss_fn(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = mia_model.state_dict()
            no_improvement = 0
        else:
            no_improvement += 1
            
        if no_improvement >= 5:
            logging.info(f"Early stopping at epoch {epoch + 1}, Best Loss: {best_loss}")
            break
        
        ## training data accuracy
        train_acc = evaluate_mia_model(mia_model, mia_data_loader, device)
        logging.info(f"Epoch {epoch + 1}, Training Accuracy: {train_acc}")        
        
    final_model = MIA().to(device)
    final_model.load_state_dict(best_model)
    return final_model

def evaluate_nn_mia_model(mia_model, data_loader, device):
    mia_model.eval()
    outputs, true_labels = get_model_outputs(mia_model, data_loader, device)
    predictions = (torch.sigmoid(outputs) > 0.5).float()
    accuracy = accuracy_score(predictions.cpu(), true_labels.cpu())
    return accuracy

def compute_metrics(eval_pred):
    """
    eval_pred is (logits, labels). We'll compute several metrics:
    Accuracy, Precision, Recall, and F1-score.
    """
    
    logits, labels = eval_pred
    
    # Check and convert logits to tensor if necessary
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
        
    # Check and convert labels to tensor if necessary
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    
    # Get Top-1 Predictions
    predictions = torch.argmax(logits, dim=-1)

    # Define torchmetrics
    accuracy_metric = Accuracy(task="multiclass", num_classes=logits.shape[1]).to(logits.device)
    top_3_accuracy_metric = Accuracy(task="multiclass", num_classes=logits.shape[1], top_k=3).to(logits.device)
    top_5_accuracy_metric = Accuracy(task="multiclass", num_classes=logits.shape[1], top_k=5).to(logits.device)
    precision_metric = Precision(task="multiclass", num_classes=logits.shape[1], average="macro").to(logits.device)
    recall_metric = Recall(task="multiclass", num_classes=logits.shape[1], average="macro").to(logits.device)
    f1_metric = F1Score(task="multiclass", num_classes=logits.shape[1], average="macro").to(logits.device)
    
    accuracy_1 = accuracy_metric(predictions, labels)
    precision = precision_metric(predictions, labels)
    recall = recall_metric(predictions, labels)
    f1 = f1_metric(predictions, labels)

    accuracy_3 = top_3_accuracy_metric(logits, labels)
    accuracy_5 = top_5_accuracy_metric(logits, labels)
    
    return {
        "accuracy_1": accuracy_1.item(),
        "accuracy_3": accuracy_3.item(),
        "accuracy_5": accuracy_5.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item()
    }
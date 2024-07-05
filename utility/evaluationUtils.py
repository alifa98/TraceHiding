import logging
import torch
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
from tqdm import tqdm

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
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            model_outputs.append(outputs)
            true_labels.append(labels)
    return torch.cat(model_outputs), torch.cat(true_labels)


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def train_mia_model(mia_data_loader, in_feature_size, device):
    
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


def evaluate_mia_model(mia_model, data_loader, device):
    mia_model.eval()
    outputs, true_labels = get_model_outputs(mia_model, data_loader, device)
    predictions = (torch.sigmoid(outputs) > 0.5).float()
    accuracy = accuracy_score(predictions.cpu(), true_labels.cpu())
    return accuracy
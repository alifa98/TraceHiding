import logging
import torch
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
from tqdm import tqdm


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
                torch.nn.Linear(in_feature_size, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
                torch.nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.model(x)
    
    mia_model = MIA().to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(mia_model.parameters(), lr=1e-4)
    
    mia_model.train()
    # early stopping with patience of 5 epochs
    best_loss = float('inf')
    best_model = None
    no_improvement = 0
    for epoch in range(100):
        pbar = tqdm(mia_data_loader, desc=f"Training MIA model, epoch {epoch + 1}", leave=False)
        for batch in pbar:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = mia_model(inputs)
            loss = loss_fn(outputs, labels.unsqueeze(1))
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
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break
        
    logging.info(f"Best Loss: {best_loss}")
    
    final_model = MIA().to(device)
    final_model.load_state_dict(best_model)
    return final_model


def evaluate_mia_model(mia_model, data_loader, device):
    mia_model.eval()
    outputs, _ = get_model_outputs(mia_model, data_loader, device)
    predictions = (outputs > 0.5).float()
    accuracy = accuracy_score(predictions.cpu(), data_loader.dataset.tensors[1].cpu())
    return accuracy
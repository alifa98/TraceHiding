import torch
from scipy.stats import entropy
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score



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
    mia_model = torch.nn.Sequential(
        torch.nn.Linear(in_feature_size, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
        torch.nn.Sigmoid()
    ).to(device)
    
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(mia_model.parameters(), lr=1e-3)
    
    mia_model.train()
    # early stopping with patience of 5 epochs
    for epoch in range(7):
        for batch in mia_data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = mia_model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    return mia_model


def evaluate_mia_model(mia_model, data_loader):
    mia_model.eval()
    outputs, _ = get_model_outputs(mia_model, data_loader)
    predictions = (outputs > 0.5).float()
    accuracy = accuracy_score(predictions.cpu(), data_loader.dataset.tensors[1].cpu())
    return accuracy
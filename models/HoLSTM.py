import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy

class LitHigherOrderLSTM(pl.LightningModule):
    def __init__(self, vocab_size, user_size, embedding_size, hidden_size, num_layers, dropout=0.2):
        super(LitHigherOrderLSTM, self).__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.user_size = user_size
        self.embeding = nn.Embedding(vocab_size + 1, embedding_size, padding_idx=0) # +1 for padding
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.linear1 = nn.Linear(hidden_size, (hidden_size + user_size)//2)
        self.linear2 = nn.Linear((hidden_size + user_size)//2, user_size)

    def forward(self, x):
        x = self.embeding(x)
        x = torch.squeeze(x, dim=2)
        x, _ = self.lstm(x)
        x = x[:, -1, :] # Get the hidden state of the last time step
        return self.linear2(F.relu(self.linear1(x))) # output the logits

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
    def training_step(self, train_batch, batch_idx):
        self.train()
        x, y = train_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        accuracy = Accuracy(task='multiclass', num_classes=self.user_size, top_k=1).to(self.device)
        acc = accuracy(y_hat, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', acc, prog_bar=True)
        return loss
    
    def test_model(self, data_loader):
        self.eval()
        accuracy_1 = Accuracy(task='multiclass', num_classes=self.user_size, top_k=1).to(self.device)
        accuracy_3 = Accuracy(task='multiclass', num_classes=self.user_size, top_k=3).to(self.device)
        accuracy_5 = Accuracy(task='multiclass', num_classes=self.user_size, top_k=5).to(self.device)
        precision = Accuracy(task='multiclass', num_classes=self.user_size, top_k=1).to(self.device)
        recall = Accuracy(task='multiclass', num_classes=self.user_size, top_k=1).to(self.device)
        f1 = Accuracy(task='multiclass', num_classes=self.user_size, top_k=1).to(self.device)
        for x, y in data_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self(x)
            accuracy_1(y_hat, y)
            accuracy_3(y_hat, y)
            accuracy_5(y_hat, y)
            precision(y_hat, y)
            recall(y_hat, y)
            f1(y_hat, y)
            
        return accuracy_1.compute(), accuracy_3.compute(), accuracy_5.compute(), precision.compute(), recall.compute(), f1.compute()
        

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
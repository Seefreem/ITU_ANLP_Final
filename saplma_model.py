import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score

# SAPLMA Classifier from https://arxiv.org/abs/2304.13734
class SaplmaClassifier(nn.Module):
    def __init__(self, input_size):
        super(SaplmaClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)



def train_classifier(classifier, train_loader, optimizer, criterion, epochs=5, device="cpu"):
    classifier.train()
    for epoch in range(epochs):
        for batch in train_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = classifier(X_batch)

            loss = criterion(outputs.flatten(), y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")



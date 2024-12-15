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



def train_classifier_saplma(classifier, train_loader, optimizer, criterion, epochs=30, device="cpu"):
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

def evaluate_classifier_saplma(classifier, test_loader, device="cpu"):
    classifier.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = classifier(X_batch)
            y_true.append(y_batch)
            y_pred.append(outputs)

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    y_pred = y_pred > 0.5
    y_pred = y_pred.flatten()
    correct_predictions = (y_pred == y_true).sum()

    accuracy = correct_predictions / len(y_true)

    print(f"Accuracy: {accuracy}")
    return accuracy




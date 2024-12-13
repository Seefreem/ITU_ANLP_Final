from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score


@variational_estimator
class BayesianSAPLMA(nn.Module):
    def __init__(self, input_size):
        super(BayesianSAPLMA, self).__init__()

        self.network = nn.Sequential(
            BayesianLinear(input_size, 256),  
            nn.ReLU(),
            BayesianLinear(256, 128),        
            nn.ReLU(),
            BayesianLinear(128, 64),         
            nn.ReLU(),
            BayesianLinear(64, 1),           
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
def train_classifier_bnn(classifier, train_loader, optimizer, criterion, epochs=5, device="cuda"):
    classifier.train()


    for epoch in range(epochs):
        for batch in train_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()           
            loss = classifier.sample_elbo(
                inputs=X_batch,
                labels=y_batch,
                criterion=lambda preds, targets: criterion(preds.squeeze(), targets),
                sample_nbr=10, 
            )
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


def evaluate_classifier_bnn(classifier, test_loader, criterion, device="cuda", num_samples=10):
    classifier.eval() 
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  
        for batch in test_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

           
            outputs = [classifier(X_batch) for _ in range(num_samples)]
            outputs = torch.stack(outputs)  
            pred_mean = outputs.mean(dim=0).flatten()  

           
            loss = criterion(pred_mean, y_batch)
            total_loss += loss.item()

            
            all_predictions.extend((pred_mean > 0.5).cpu().numpy()) 
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_predictions)

    return avg_loss, accuracy

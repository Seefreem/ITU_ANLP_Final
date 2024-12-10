import torch
from model_utils import load_model_and_tokenizer, extract_hidden_states, split_data
from saplma_model import SaplmaClassifier, train_classifier, evaluate_classifier
from dataset_scripts.load_data import extract_hidden_states_with_labels
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main( hidden_states_file, labels_file):
    # Load data
    hidden_states, labels = extract_hidden_states_with_labels(hidden_states_file,labels_file) # TODO: Ask chatgpt to label data, if short_answer matches response, label is 1, else 0

    input_size = 4096 #len(hidden_states[0])
    classifier = SaplmaClassifier(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    X_train, X_test, y_train, y_test = split_data(hidden_states, labels) # TODO: Implement split_data
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)


    train_classifier(classifier, train_loader, optimizer, criterion, device=device)

    evaluate_classifier(classifier, test_loader, device=device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_states_path", type=str, required=True)
    parser.add_argument("--labels_file", type=str, required=True)
    args = parser.parse_args()
    print(f"Using device: {device}")
    hidden_states = args.hidden_states_path
    labels = args.labels_file

    main(hidden_states, labels)
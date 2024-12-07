import torch
from model_utils import load_model_and_tokenizer, extract_hidden_states
from saplma_model import SaplmaClassifier, train_classifier
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import argparse



def main(access_token, model_name, dataset_file):
    # Load data
    hidden_states, labels = load_hiddens_states_labels(dataset_file) # TODO: Ask chatgpt to label data, if short_answer matches response, label is 1, else 0

    input_size = len(hidden_states[0])
    classifier = SaplmaClassifier(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    X_train, X_test, y_train, y_test = split_data(hidden_states, labels) # TODO: Implement split_data
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=8)

    # Train classifier
    train_classifier(classifier, train_loader, optimizer, criterion, device=device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--filepath", type=str, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_name = args.model #"PY007/TinyLlama-1.1B-step-50K-105b"
    dataset_file = args.filepath

    main(args.token, args.model, args.filepath)
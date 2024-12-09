import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()
    return model, tokenizer

@torch.no_grad()
def extract_hidden_states(model, inputs, layer_index):
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states[layer_index]

    return hidden_states.mean(dim=1) # Single vector per input in the batch that represents the entire sentence



def split_data(hidden_states, labels, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        hidden_states.numpy(), labels, test_size=test_size, random_state=random_state
    )
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

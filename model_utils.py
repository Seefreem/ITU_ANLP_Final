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

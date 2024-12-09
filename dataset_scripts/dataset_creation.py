import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from load_data import load_and_prepare_data, create_csv


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def initialize_model(model_name, token):

    os.environ["HUGGING_FACE_TOKEN"] = token
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.pad_token_id
    )
    return tokenizer, model


def generate_response(tokenizer, model, question, max_new_tokens=6, layer_step=5):

    # Tokenize the input question and feed them into the model
    inputs = tokenizer(question, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=True
    )

    # Extract hidden states of every 4th layer - 0, 4, 8, 12, 16 and etc.
    hidden_states = []
    total_layers = len(outputs.hidden_states) #16 - 20 - 24 - 28 - 32

    print(f"Total layers: {total_layers}")


    hidden_state = outputs.hidden_states[1][16].mean(dim=1)
    hidden_states.append(hidden_state.squeeze().tolist())
    hidden_state = outputs.hidden_states[1][20].mean(dim=1)
    hidden_states.append(hidden_state.squeeze().tolist())



    # Decode the generated sequence
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    generated_response = generated_text.split(question, 1)[-1].strip()

    return generated_response, hidden_states


def main(access_token, model_name):

    # Load and prepare data
    ids, questions, references = load_and_prepare_data("../data/natural_questions_sample.csv")

    # Initialize models
    tokenizer, model = initialize_model(model_name, access_token)

    # Generate and evaluate responses
    responses = []
    hidden_states = []

    for i, question in enumerate(questions):
        # Generate a response
        response, states  = generate_response(tokenizer, model, question)
        responses.append(response)
        hidden_states.append(states)
        print(f"Question {i+1}: {question}")


    # Create a CSV file with the data
    create_csv(ids, questions, references, responses, hidden_states)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args.token, args.model)

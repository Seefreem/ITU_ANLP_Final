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
        output_hidden_states=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract hidden states of every 4th layer - 0, 4, 8, 12, 16 and etc.
    hidden_states = []
    total_layers = len(outputs.hidden_states[1]) #16 - 20 - 24 - 28 - 32


    #for i in range(0, total_layers, layer_step):
        #hidden_state = outputs.hidden_states[1][i+1].mean(dim=1)
        #hidden_states.append(hidden_state.squeeze().tolist())

    hidden_state = outputs.hidden_states[1][16].mean(dim=1)
    hidden_states.append(hidden_state.squeeze().tolist())

    # Decode the generated sequence
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    generated_response = generated_text.split(question, 1)[-1].strip()

    return generated_response, hidden_states, outputs


def main(access_token, model_name,file_path):

    # Load and prepare data
    questions, references = load_and_prepare_data(file_path)

    # Initialize models
    tokenizer, model = initialize_model(model_name, access_token)
    length_set = set()
    for i in references:
        length_set.add(len(tokenizer.tokenize(i)))
    print(length_set, max(length_set))
    # Generate and evaluate responses
    responses = []
    hidden_states = []
    logits_of_answers = []

    for i, question in enumerate(questions):

        # Generate a response
        response, states, outputs = generate_response(tokenizer, model, question)
        responses.append(response)
        hidden_states.append(states)

        # logits extraction
        question_length = len(tokenizer(question)['input_ids'])
        logits = torch.nn.functional.softmax(torch.cat(outputs.scores,0), dim=1)
        average_probability = 0

        for j in range(len(outputs[0][0][question_length:])):
            average_probability += logits[j, outputs[0][0][question_length:][j]]
        average_probability /= 6
        logits_of_answers.append(average_probability.item())

        # Print the question
        print(f"Question {i+1}: {question} {response}; GT: {references[i]}")

    # Create a CSV file with the data
    create_csv(questions, references, responses, hidden_states, logits_of_answers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--filepath", type=str, required=True)
    args = parser.parse_args()
    main(args.token, args.model, args.filepath)

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
import argparse
from dataset_scripts.load_data import load_and_prepare_data, create_csv
from dataset_scripts.dataset_creation import initialize_model, generate_response

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def main(access_token, model_name,file_path):

    # Load and prepare data
    questions, references = load_and_prepare_data(file_path)

    # Initialize models
    tokenizer, model = initialize_model(model_name, access_token)

    # Generate and evaluate responses
    logits_of_answers = []
    responses = []
    for i, question in enumerate(questions):
        # Generate a response
        response, states, outouts  = generate_response(tokenizer, model, question)
        responses.append(response)
        # logits.append(response)
        question_lenth = len(tokenizer(question)['input_ids'])
        # print(tokenizer.decode(outouts[0][0][question_lenth:]))
        logits = torch.nn.functional.softmax(torch.cat(outouts.scores,0), dim=1)
        # print(f"logits : {logits.shape}")
        # print(f"logits : {torch.sum(logits, 1)}")
        average_probability = 0
        for i in range(len(outouts[0][0][question_lenth:])):
            # print(logits[i, outouts[0][0][question_lenth:][i]])
            average_probability += logits[i, outouts[0][0][question_lenth:][i]]
        average_probability /= 6
        # print(average_probability)
        logits_of_answers.append(average_probability.item())

    dict = {'questions': questions, 'short_answers': references, 
            'responses': responses, 'logits_of_answers': logits_of_answers} # load_data.py create_csv!!!!!
    # Create a CSV file with the data
    df = pd.DataFrame(dict)
    # saving the dataframe
    df.to_csv('dataset_with_logits.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--filepath", type=str, required=True)
    args = parser.parse_args()
    main(args.token, args.model, args.filepath)

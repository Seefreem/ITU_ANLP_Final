import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from load_data import load_and_prepare_data

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def initialize_model(model_name, token):

    os.environ["HUGGING_FACE_TOKEN"] = token
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    return tokenizer, model

def generate_response(tokenizer, model, question, max_new_tokens=10):

    inputs = tokenizer(question, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True
    )

    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    return generated_text

def evaluate_response(tokenizer, model, question, response, reference=None, max_new_tokens=10):

    prompt = f"""
    Evaluate the following response for correctness and relevance, answer with single word 'Yes' or 'No':
    Question: {question}
    Response: {response}
    """
    if reference:
        prompt += f"\nReference Answer: {reference}"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True
    )

    decoded_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    return decoded_text.strip()

def split_questions_by_evaluation(questions, evaluations):

    correct = []
    incorrect = []
    for question, evaluation in zip(questions, evaluations):
        if evaluation.strip().lower().startswith('yes'):
            correct.append(question)
        else:
            incorrect.append(question)
    return correct, incorrect

def main(access_token, model_name):
    # Load and prepare data
    questions, references = load_and_prepare_data()

    # Initialize models
    tokenizer, model = initialize_model(model_name, access_token)

    # Generate and evaluate responses
    responses = []
    evaluations = []

    for i, question in enumerate(questions):
        # Generate a response
        response = generate_response(tokenizer, model, question)
        responses.append(response)

        # Evaluate the response
        reference = references[i] if references else None
        evaluation = evaluate_response(tokenizer, model, question, response, reference=reference)
        evaluations.append(evaluation)

        #print(f"Question {i+1}: {question}")
        #print(f"Response: {response}")
        #print(f"Evaluation: {evaluation})")
        #print("-" * 30)

    # Split questions by evaluation
    correct, incorrect = split_questions_by_evaluation(questions, evaluations)

    # Display results
    print("Correctly Answered Questions:", correct)
    print("Incorrectly Answered Questions:", incorrect)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args.token, args.model)

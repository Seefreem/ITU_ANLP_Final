import os
from transformers import AutoTokenizer, pipeline
import torch
import argparse
import transformers
from load_data import load_and_prepare_data


def initialize_huggingface_model(token, model_name="meta-llama/Llama-3.1-8B-Instruct"):

    os.environ["HUGGING_FACE_TOKEN"] = token
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=token
    )

    return pipeline


def split_into_right_wrong(questions, evaluations):
    correctly_answered = []
    incorrectly_answered = []

    for question, evaluation in zip(questions, evaluations):
        first_word = evaluation.split()[0].lower()
        if 'yes' in first_word:
            correctly_answered.append(question)
        elif 'no' in first_word:
            incorrectly_answered.append(question)
        else:
            incorrectly_answered.append(question)

    return correctly_answered, incorrectly_answered



def generate_answers_with_huggingface(pipeline, questions):

    responses = []
    for question in questions:

        outputs = pipeline(
            question,
            max_new_tokens=10,
            return_full_text=False,
            num_return_sequences=1,
        )

        responses.append(outputs[0]['generated_text'])

        print(f"Question: {question}")
        print(f"Answer: {outputs[0]['generated_text']}")
        print("-" * 30)
    return responses

# Evaluate answers using the second model
def evaluate_responses_with_huggingface(pipeline, questions, responses, references=None):
    evaluations = []

    for i, (question, response) in enumerate(zip(questions, responses)):
        prompt = f"""
        Evaluate the following response for correctness and relevance, answer with single word 'Yes' or 'No':

        Question: {question}
        Response: {response}
        """
        if references:
            prompt += f"\nReference Answer: {references[i]}"



        # Generate the evaluation
        outputs = pipeline(
            prompt,
            max_new_tokens=25,
            num_return_sequences=1,
        )

        evaluations.append(outputs[0]['generated_text'])


    return evaluations


# Main function
def evaluate_model(access_token):

    # Load and prepare data
    questions, references = load_and_prepare_data()

    # Initialize models
    answer_generator_model = initialize_huggingface_model(access_token, model_name=model_name)
    evaluator_model = initialize_huggingface_model(access_token, model_name=model_name)

    # Generate responses
    responses = generate_answers_with_huggingface(answer_generator_model,  questions)

    # Evaluate responses
    evaluations = evaluate_responses_with_huggingface(
        pipeline=evaluator_model,
        questions=questions,
        responses=responses,
        references=references
    )


    correct, incorrect = split_into_right_wrong(questions, evaluations)

    print("Correctly Answered Questions: ", correct)
    print("Incorrectly Answered Questions: ", incorrect)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--model_evaluator", type=str, required=True)
    parser.add_argument("--model_generator", type=str, required=False)
    args = parser.parse_args()
    model_name = args.model_evaluator
    access_token = args.token
    evaluate_model(access_token)


#TODO: Run evaluation pipeline on HPC
#TODO: Use batches for efficiency
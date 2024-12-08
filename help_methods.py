def split_questions_by_evaluation(questions, evaluations):

    correct = []
    incorrect = []
    for question, evaluation in zip(questions, evaluations):
        if evaluation.strip().lower().startswith('yes'):
            correct.append(question)
        else:
            incorrect.append(question)
    return correct, incorrect



def evaluate_response(tokenizer, model, question, response, reference=None, max_new_tokens=4):
    prompt = f"""
    You are tasked with evaluating the response for its correctness and relevance to the given question. Answer with a single word: 'Yes' if the response is correct and relevant, or 'No' otherwise. Do not provide any additional explanation.

    Question: {question}
    Response: {response}
    """
    if reference:
        prompt += f"\nCorrect Answer: {reference}"

    prompt += "\nYour evaluation:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True
    )

    decoded_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    evaluation_result = decoded_text.strip().split("Your evaluation:")[-1].strip()

    return evaluation_result
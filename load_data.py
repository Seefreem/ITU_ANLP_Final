
import json
import os
import deeplake
def load_and_prepare_data(link='hub://activeloop/squad-val'):
    ds = deeplake.load(link)
    dataloader = ds.pytorch(num_workers=0, batch_size=4, shuffle=True)
    questions = []
    references = []
    answers = []

    # Iterate through the dataloader batches
    for batch in dataloader:
        for i in range(len(batch['question'])):
            questions.append(batch['question'][i])
            references.append(batch['context'][i])
            answers.append(batch['text'][i])
        if len(questions) >= 12:  # For demonstation purposes, stop after 40 questions
            break

    return questions, answers


# https://datasets.activeloop.ai/docs/ml/datasets/squad-dataset/ - SQuAD dataset loading and preparation
# I am not sure if it is the best way to extract the data from this dataset, but it worked for me.
# I am not sure how to use batches to efficiently load data
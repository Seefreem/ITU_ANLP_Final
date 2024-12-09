
import json
import os
import csv
import pandas as pd
def load_and_prepare_data(file_path): # assume it is csv data # Create empty lists to store the extracted data
    ids = []
    questions = []
    short_answers = []

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Iterate over the rows and append each relevant field to the lists
        for row in reader:
            ids.append(row['ID'])
            questions.append(row['question'])
            short_answers.append(row['short_answers'])

    return ids, questions, short_answers

def load_and_prepare_data(file_path): # assume it is csv data # Create empty lists to store the extracted data

    questions = []
    short_answers = []

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Iterate over the rows and append each relevant field to the lists
        for row in reader:
            questions.append(row['question'])
            short_answers.append(row['short_answers'])

    return questions, short_answers



def create_csv(questions, short_answers, responses, hidden_states):
    # dictionary of lists
    dict = {'questions': questions, 'short_answers': short_answers, 'responses': responses}

    df_chatgpt4 = pd.DataFrame(dict)
    df_chatgpt4.to_csv('chatgpt4_evaluation.csv')

    dict[f'hidden_state_{16}'] = hidden_states
    df = pd.DataFrame(dict)

    # saving the dataframe
    df.to_csv('dataset_training.csv')



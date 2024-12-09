
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



def create_csv(ids, questions, short_answers, responses, hidden_states):
    # dictionary of lists
    dict = {'IDs': ids, 'questions': questions, 'short_answers': short_answers, 'responses': responses}

    size = len(hidden_states[0])

    for i in range(size):
        dict[f'hidden_state_{i}'] = [states[i] for states in hidden_states]


    df = pd.DataFrame(dict)

    # saving the dataframe
    df.to_csv('dataset.csv')



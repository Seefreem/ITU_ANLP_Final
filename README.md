# ITU_ANLP_Final
This is the repo of the course project of Advanced Natural Language Processing and Deep Learning (Autumn 2024) at IT University of Copenhagen.

# Group Name and Members
Group 8 bilibili: Shiling Deng, Ivan Rozhdestvenskii and Levente András Wallis.


# Create two datasets: chatgpt4_evaluation.csv and dataset_training.csv
chatgpt4_evaluation.csv - contains the questions, short answers (gt) and responses from the pretrained model - for evaluation purposes
dataset_training.csv - contains the questions, short answers (gt), hidden layers representations for training purposes

Example command:
```shell
python dataset_creation.py  --token gt_dasdasdasdasdsafrgwr --model google/gemma-2-2b-it --filepath data/natural_questions_sample.csv
```


# Train a SAPLMA/BayseianSAPLMA model for classification of the questions whether the model is able to answer them
Example command:
-hidden_state_path - csv file with hidden states
-labels_file - csv file with labels evaluated by chatgpt4
-arc - architecture of the model (saplma or bnn)
```shell
python classifier_train.py  --hidden_states_path dataset_training.csv --labels_file Evaluation_of_Responses.csv --arc saplma/bnn
```


# Get the averaged probabilities of answers - INTEGRATED INTO dataset_creation.py
Example command:
```shell
python logits.py  --token google/gemma-2-2b-it --model google/gemma-2-2b-it --filepath data/natural_questions_sample.csv
```
 

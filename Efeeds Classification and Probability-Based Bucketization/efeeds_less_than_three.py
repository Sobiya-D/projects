import pandas as pd
import numpy as np
import joblib
from nltk.stem import PorterStemmer
import spacy
from spacy.lang.en import STOP_WORDS
import re
from ftfy import fix_encoding
import logging
import json
import os

# Read configuration from config.json
with open('config.json') as config_file:
    config = json.load(config_file)

# Update file paths
MODEL_PATH_1 = config["MODEL_PATH_1"]
MODEL_PATH_2 = config["MODEL_PATH_2"]
INPUT_PATH = config["INPUT_PATH"]
OUTPUT_PATH = config["OUTPUT_PATH"]

# Loading models using updated paths
def load_models():
    vectorizer = joblib.load(os.path.join(MODEL_PATH_1, 'VEC_11_01_2024.pkl'))
    model = joblib.load(os.path.join(MODEL_PATH_1, 'SGD_11_01_2024.pkl'))
    scaler = joblib.load(os.path.join(MODEL_PATH_1, 'scaler_11_01_2024.pkl'))

    t_vectorizer = joblib.load(os.path.join(MODEL_PATH_2, 'T194_VEC_25_01_2024.pkl'))
    t_model = joblib.load(os.path.join(MODEL_PATH_2, 'T194_SGD_25_01_2024.pkl'))
    t_scaler = joblib.load(os.path.join(MODEL_PATH_2, 'T104_scaler_25_01_2024.pkl'))
    return vectorizer, model, scaler, t_vectorizer, t_model, t_scaler

vectorizer, model, scaler, t_vectorizer, t_model, t_scaler = load_models()

# Function to filter out invalid indices for second model
def t_general_filter(dataframe):
    t_unique_indices = set()
    t_invalid_indices = dataframe[dataframe.Token_Ct < 5].index.tolist()
    t_unique_indices.update(t_invalid_indices)
    return list(t_unique_indices)

# Function to perform predictions for further processing
def t_preds(tdf):
    t_vals = t_vectorizer.transform(tdf.DESCRIPTION)
    t_vals = t_scaler.transform(t_vals)
    tz = t_model.predict_proba(t_vals)
    t_probas = [max(i) for i in tz]
    t_labels = np.argmax(tz, axis=1)
    t_classes = t_model.classes_
    t_labels_classes = [t_classes[i] for i in t_labels]
    return t_probas, t_labels_classes

def stemmer(sentence):
    ps = PorterStemmer()
    return ' '.join(ps.stem(token) for token in sentence.split())

def fix_sentence(sentence):
    sentence = fix_encoding(sentence)
    sentence = re.sub(r'http\S+', ' ', sentence)
    sentence = re.sub(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', ' ', sentence)
    sentence = re.sub(r'[^a-z\s+]', ' ', sentence.lower())
    sentence = stemmer(sentence)
    final_sentence = []
    for token in sentence.split():
        if token not in STOP_WORDS and len(token) >= 3:
            final_sentence.append(token)
    if len(final_sentence) >= 3:
        return re.sub(' +', ' ', ' '.join(final_sentence)).strip()
    else:
        return ''

# Function to bucketize labels based on score
def bucketizing_logic(label, score):
    score *= 100
    if score >= 99:
        return f'{label} - High 1'
    elif 96 <= score <= 98.9:
        return f'{label} - High 2'
    elif 91 <= score <= 95.9:
        return f'{label} - High 3'
    elif 81 <= score <= 90.9:
        return f'{label} - Medium 1'
    elif 76 <= score <= 80.9:
        return f'{label} - Medium 2'
    else:
        return f'{label} - Low'

# Function to filter out invalid indices
def general_filter(dataframe):
    unique_indices = set()
    invalid_indices = dataframe[dataframe.Token_Ct < 5].index.tolist()
    unique_indices.update(invalid_indices)
    return list(unique_indices)

# Function to transform labels
def transform_labels(dataframe):
    # Create a new column "t_Final_label" based on the values in the "t_class_name" column
    dataframe['Final_label'] = dataframe['t_class_name'].apply(lambda x: 'Valid - Low 1' if x == 'Valid' else 'Invalid - High 1')
    return dataframe

# Function to combine DataFrames
def combine_dataframes(invalid_high, model_1_output):
    invalid_high = invalid_high.drop(columns=['t_class_name'])
    combined_model_output = pd.concat([invalid_high, model_1_output], ignore_index=True)
    return combined_model_output

def preds(df, vectorizer, scaler, model):
    vals = vectorizer.transform(df['DESCRIPTION'])
    vals = scaler.transform(vals)
    z = model.predict_proba(vals)
    probas = [max(i) for i in z]
    labels = np.argmax(z, axis=1)
    classes = model.classes_
    labels_classes = [classes[i] for i in labels]
    return probas, labels_classes, z[:, 0], z[:, 1]  # Return probabilities for both classes

def main_logic(input_data, preprocessed_data):
    try:
        preprocessed_data = preprocessed_data.fillna('')
        preprocessed_data['Token_Ct'] = preprocessed_data['DESCRIPTION'].map(lambda x: len(x.split()))
        preprocessed_data['DESCRIPTION'] = preprocessed_data['DESCRIPTION'].map(lambda x: fix_sentence(x))
        invalid_indices = general_filter(preprocessed_data)
        labels = {val: 'Invalid - Low' for val in invalid_indices}
        filtered_df = preprocessed_data.loc[list(set(preprocessed_data.index) - set(invalid_indices))]

        # Perform predictions and update filtered_df
        probas, class_names, invalid_probs, valid_probs = preds(filtered_df, vectorizer, scaler, model)
        filtered_df['probs'], filtered_df['class_name'], filtered_df['Invalid_prediction_Probability'], filtered_df['Valid_prediction_Probability'] = probas, class_names, invalid_probs, valid_probs

        # Calculate Final_label and update input_data
        final_labels = [bucketizing_logic(row['class_name'], row['probs']) for _, row in filtered_df.iterrows()]
        labels2 = {index: final_label for index, final_label in zip(filtered_df.index, final_labels)}
        final_labels_df = {**labels, **labels2}
        final_labels_df = dict(sorted(final_labels_df.items())).values()
        input_data['Final_label'] = final_labels_df

        # Iterate over rows in input_data and update probabilities and class names
        for index, row in input_data.iterrows():
            input_id = row['Excel_Unique_ID']
            if input_id in filtered_df['Excel_Unique_ID'].values:
                filtered_row = filtered_df[filtered_df['Excel_Unique_ID'] == input_id].iloc[0]
                input_data.at[index, 'probs'] = filtered_row['probs']
                input_data.at[index, 'class_name'] = filtered_row['class_name']
                input_data.at[index, 'Invalid_prediction_Probability'] = filtered_row['Invalid_prediction_Probability']
                input_data.at[index, 'Valid_prediction_Probability'] = filtered_row['Valid_prediction_Probability']
            else:
                input_data.at[index, 'probs'] = 0.0
                input_data.at[index, 'class_name'] = 'Invalid'
                input_data.at[index, 'Invalid_prediction_Probability'] = 0.0
                input_data.at[index, 'Valid_prediction_Probability'] = 0.0

        # Separate rows with Final_label 'Invalid - High 1' and 'Invalid - High 1' from input_data
        invalid_high = input_data[input_data['Final_label'] == 'Invalid - High 1'].copy()
        model_1_output = input_data[input_data['Final_label'] != 'Invalid - High 1'].copy()

        # Drop 'Final_label' column from invalid_high
        invalid_high.drop(columns=['Final_label'], inplace=True)

        # Further processing for invalid_high if not empty
        if not invalid_high.empty:
            t_preprocessed_data = invalid_high.copy()
            t_preprocessed_data = t_preprocessed_data.fillna('')
            t_preprocessed_data['Token_Ct'] = t_preprocessed_data['DESCRIPTION'].map(lambda x: len(x.split()))
            t_preprocessed_data['DESCRIPTION'] = t_preprocessed_data['DESCRIPTION'].map(lambda x: fix_sentence(x))
            t_invalid_indices = t_general_filter(t_preprocessed_data)
            t_filtered_df = t_preprocessed_data.loc[list(set(t_preprocessed_data.index) - set(t_invalid_indices))]
            t_labels = {val: 'Invalid - Low' for val in t_invalid_indices}
            t_filtered_df['t_probs'], t_filtered_df['t_class_name'] = t_preds(t_filtered_df)
            invalid_high = invalid_high.merge(t_filtered_df[['Excel_Unique_ID', 't_class_name']], on='Excel_Unique_ID', how='left')
            invalid_high = transform_labels(invalid_high)
            combined_model_output = combine_dataframes(invalid_high, model_1_output)
        else:
            combined_model_output = model_1_output.copy()

        return combined_model_output

    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

def main():
    try:
        # Iterate over all files in the input folder
        for filename in os.listdir(INPUT_PATH):
            if filename.endswith('.txt'):
                input_file_path = os.path.join(INPUT_PATH, filename)

                # Read the text file into DataFrame with 'latin1' encoding
                df = pd.read_csv(input_file_path, delimiter='<TAB>', encoding='latin1', engine='python')

                input_data = df.copy()
                logging.info(f'Loaded input data with {len(input_data)} rows.')

                # Check if the expected columns are present in the DataFrame
                expected_columns = ['Excel_Unique_ID', 'DESCRIPTION']
                if not all(col in input_data.columns for col in expected_columns):
                    raise ValueError(f"Input DataFrame does not contain all expected columns: {expected_columns}")

                preprocessed_data = input_data.copy()
                combined_model_output = main_logic(input_data, preprocessed_data)

                # Rename columns and set output file name based on input file name
                combined_model_output.rename(columns={'Excel_Unique_ID': 'ID'}, inplace=True)
                output_file_name = filename  # Output file name is same as input file name

                # Iterate over rows in combined_model_output and update probabilities
                for index, row in combined_model_output.iterrows():
                    if row['Final_label'] == 'Valid - Low 1':
                        combined_model_output.at[index, 'Invalid_prediction_Probability'] = 0.30
                        combined_model_output.at[index, 'Valid_prediction_Probability'] = 0.70

                # Create Predicted_Flag column based on Final_label
                combined_model_output['Predicted_Flag'] = combined_model_output['Final_label'].apply(lambda x: 1 if x.startswith('Valid') else 0)

                # Select only desired columns
                combined_model_output = combined_model_output[['ID', 'Predicted_Flag', 'Invalid_prediction_Probability', 'Valid_prediction_Probability']]

                # Save output to a text file in the output folder
                # Create the output directory if it doesn't exist
                if not os.path.exists(OUTPUT_PATH):
                    os.makedirs(OUTPUT_PATH, exist_ok=True)
                    
                output_file_path = os.path.join(OUTPUT_PATH, output_file_name)

                combined_model_output.to_csv(output_file_path, sep='\t', index=False)

    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

# Execute the main function
main()
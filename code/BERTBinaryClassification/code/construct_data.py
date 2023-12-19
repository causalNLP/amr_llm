import pandas as pd
import os
import re
import json
from sklearn.metrics import classification_report
import numpy as np
import ast
import argparse
from bleu import list_bleu
import random
from efficiency.function import set_seed
from pathlib import Path
import pdb
import math
from tqdm import tqdm



np.random.seed(0)
random.seed(0)
set_seed(0)
root_dir = Path(__file__).parent.parent.parent.parent.resolve()
parent_dir = os.path.dirname(root_dir)
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
bert_date_dir = root_dir / "code/BERTBinaryClassification/data"


def process_data(df):
    texts = pd.read_csv(data_dir / 'classifier_inputs/updated_data_input - classifier_input.csv')
    df['input'] = ''
    old_data_for_bert = pd.read_csv(bert_date_dir/'data_for_bert.csv')

    for idx, row in tqdm(df.iterrows()):
        old_row = old_data_for_bert.loc[old_data_for_bert['id'] == row['id']]
        if old_row.shape[0] == 0:
            texts_row = texts.loc[texts['id'] == row['id']]
            if 'paws' in row['id']:
                premise = extract_value2(texts_row['input_json'], 'premise')
                hypothesis = extract_value2(texts_row['input_json'], 'premise')
                df.loc[idx, 'input'] = premise + ' ' + hypothesis
            elif 'newstest' in row['id']:
                df.loc[idx, 'input'] = extract_value2(texts_row['input_json'].values[0], 'en')
            elif 'pubmed' in row['id']:
                df.loc[idx, 'input'] = extract_value2(texts_row['input_json'], 'sentence')
            elif 'logic' in row['id']:
                try:
                    df.loc[idx, 'input'] = extract_value2(texts_row['input_json'].values[0], 'source_article')
                except Exception as e:
                    print(e)


            elif 'spider' in row['id']:
                df.loc[idx, 'input'] = extract_value2(texts_row['input_json'], 'question')
        else:
            df.loc[idx, 'input'] = old_row['input'].values[0]
    df = df[~df['input'].isna()]
    return df


def extract_value2(json_str, key):
    try:
        json_data = ast.literal_eval(json_str)
        return json_data[key]
    except (json.JSONDecodeError, KeyError):
        try:
            json_data = json.loads(json_str)
            return json_data[key]
        except Exception as e:
            return None


def main():
    features = pd.read_csv(data_dir/'all_data_features.csv')
    df = features[['id','helpfulness', 'split']]
    df_with_text = process_data(df)
    df_with_text['label'] = df_with_text['helpfulness'].apply(lambda x: 1 if x > 0 else 0)
    df_with_text.to_csv(bert_date_dir/'data_for_bert_1219.csv', index=False)

if __name__ == '__main__':
    main()

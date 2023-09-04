import pandas as pd
import os
import re
import json
import numpy as np
from pathlib import Path
import math
import ast
from efficiency.function import shell

root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
out_dir = data_dir / "outputs"
parent_dir = os.path.dirname(root_dir)

data_file = data_dir / "classifier_inputs/updated_data_input - classifier_input.csv"
amr_file = data_dir / "corrected_amrs.csv"


def clean_amr(amr):
    if not isinstance(amr, str):
        amr = str(amr)
    amr = re.sub("~e.\d+", "", amr)
    return amr

def extract_value(json_str, key):
    try:
        json_data = ast.literal_eval(json_str)
        return json_data[key]
    except (json.JSONDecodeError, KeyError):
        return None



def extract_value2(json_str, key):
    try:
        json_data = json.loads(json_str)
        return json_data[key]
    except (json.JSONDecodeError, KeyError):
        return None


def process_2_clauses(df, amr):
    amr = amr.rename(columns={'id': 'id_total'})
    amr['id'] = amr.id_total.str[:-2]
    df = df.merge(amr, how='inner', on='id')
    df['id_type'] = df.id_total.str[-1:]
    df['premise'] = df['input_json'].apply(lambda x: extract_value(x, 'premise'))
    df['hypothesis'] = df['input_json'].apply(lambda x: extract_value(x, 'hypothesis'))
    df = df.pivot(index=['id', 'ground_truth', 'premise', 'hypothesis'], columns=['id_type'], values=['amr'])
    df = df.reset_index()
    # Drop a level in column names and concatenate the remaining levels
    separator = "_"
    new_columns = df.columns.get_level_values(1).astype(str)
    new_columns = df.columns.get_level_values(0) + separator + new_columns

    # Assign the modified column names to the DataFrame
    df.columns = [c.rstrip('_') for c in new_columns]
    return df


def process_data(file_path, file_path_amr, dataset, test_only = True):
    df = pd.read_csv(file_path)
    amr = pd.read_csv(file_path_amr)
    amr['amr']=amr['amr'].replace(r'\s+', ' ', regex=True)
    if 'gold' in dataset:
        df = df.loc[df.id.str.contains(dataset.replace('_gold', ''))]
        amr = amr.loc[amr.id.str.contains(dataset.replace('_gold', ''))]
    else:
        df = df.loc[df.id.str.contains(dataset)]
        amr = amr.loc[amr.id.str.contains(dataset)]
    df = df.loc[:, ['id', 'input_json', 'ground_truth']]
    if dataset in ['paws']:
        df = process_2_clauses(df, amr)
        # Save column 'premise' and 'hypothesis' to two .txt files, where is each line is a sentence, name the file according to {dataset}_{column}.txt
        df['premise'].to_csv(data_dir / 'txt_files/paws_premise.txt', index=False, header=False)
        df['hypothesis'].to_csv(data_dir / 'txt_files/paws_hypothesis.txt', index=False, header=False)
        # Save columns 'amr_p' and 'amr_h' to two .txt files, where is each line is a AMR, name the file according to {dataset}_{column}.txt
        df['amr_p'].to_csv(data_dir / f'txt_files/{dataset}_amr_p.txt', index=False, header=False)
        df['amr_h'].to_csv(data_dir / f'txt_files/{dataset}_amr_h.txt', index=False, header=False)
    elif dataset in ['django']:
        df['text'] = df['input_json'].apply(lambda x: extract_value(x, 'nl'))
        df = df.merge(amr, how='inner', on='id')
        df = df.loc[:, ['id', 'ground_truth', 'text', 'amr']].drop_duplicates()
        # Save column 'text' to a .txt file, where is each line is a sentence
        df['text'].to_csv(data_dir / f'txt_files/{dataset}.txt', index=False, header=False)
        # Save column 'amr' to a .txt file, where is each line is a sentence
        df['amr'].to_csv(data_dir / f'txt_files/{dataset}_amr.txt', index=False, header=False)
    elif dataset in ['logic']:
        df['text'] = df['input_json'].apply(lambda x: extract_value(x, 'source_article'))
        df = df.merge(amr, how='inner', on='id')
        # Save column 'text' to a .txt file, where is each line is a sentence
        df['text'].to_csv(data_dir / f'txt_files/logic.txt', index=False, header=False)
        # Save column 'amr' to a .txt file, where is each line is a sentence
        df['amr'].to_csv(data_dir / f'txt_files/{dataset}_amr.txt', index=False, header=False)

    elif dataset in ['spider']:
        df['text'] = df['input_json'].apply(lambda x: extract_value(x, 'question'))
        df = df.merge(amr, how='inner', on='id')
        # Save column 'text' to a .txt file, where is each line is a sentence
        df['text'].to_csv(data_dir / f'txt_files/{dataset}.txt', index=False, header=False)
        # Save column 'amr' to a .txt file, where is each line is a sentence
        df['amr'].to_csv(data_dir / f'txt_files/{dataset}_amr.txt', index=False, header=False)

    elif dataset in ['entity_recog_gold']:
        gold = pd.read_csv(data_dir/'ldc_ner_features_true.csv')
        gold = gold[['id', 'true_amr']]
        gold['true_amr'] = gold['true_amr'].apply(lambda x: clean_amr(x))
        df = df.merge(gold, how='inner', on='id')
        df['text'] = df['input_json'].apply(lambda x: extract_value(x, 'text'))
        df = df.merge(amr, how='inner', on='id')
        # Save column 'text' to a .txt file, where is each line is a sentence
        df['text'].to_csv(data_dir / f'txt_files/{dataset}.txt', index=False, header=False)
        # Save column 'amr' to a .txt file, where is each line is a sentence
        df['true_amr'].to_csv(data_dir / f'txt_files/{dataset}_amr.txt', index=False, header=False)
    elif dataset in ['entity_recog']:
        df['text'] = df['input_json'].apply(lambda x: extract_value(x, 'text'))
        df = df.merge(amr, how='inner', on='id')
        # Save column 'text' to a .txt file, where is each line is a sentence
        df['text'].to_csv(data_dir / f'txt_files/{dataset}.txt', index=False, header=False)
        # Save column 'amr' to a .txt file, where is each line is a sentence
        df['amr'].to_csv(data_dir / f'txt_files/{dataset}_amr.txt', index=False, header=False)

    elif dataset in ['newstest']:
        df['text'] = df['input_json'].apply(lambda x: extract_value(x, 'en'))
        df['ground_truth'] = df['input_json'].apply(lambda x: extract_value(x, 'de'))
        amr['id'] = amr['id'].str[:-3]
        df = df.merge(amr, how='inner', on='id')
        # Save column 'text' to a .txt file, where is each line is a sentence
        df['text'].to_csv(data_dir / f'txt_files/{dataset}.txt', index=False, header=False)
        # Save column 'amr' to a .txt file, where is each line is a sentence
        df['amr'].to_csv(data_dir / f'txt_files/{dataset}_amr.txt', index=False, header=False)
    elif dataset in ['pubmed']:
        df['text'] = df['input_json'].apply(lambda x: extract_value(x, 'sentence'))
        df['interaction'] = df['input_json'].apply(lambda x: extract_value(x, 'interaction'))
        df = df.merge(amr, how='inner', on='id')
        # Save column 'text' to a .txt file, where is each line is a sentence
        df['text'].to_csv(data_dir / f'txt_files/{dataset}.txt', index=False, header=False)
        # Save column 'amr' to a .txt file, where is each line is a sentence
        df['amr'].to_csv(data_dir / f'txt_files/{dataset}_amr.txt', index=False, header=False)
    elif dataset in ['ldc_dev']:
        amr = amr.assign(id_type=np.where(amr.id.str.endswith('nonpara'), 'nonpara',
                                          np.where(amr.id.str.endswith('para'), 'para',
                                                   np.where(amr.id.str.endswith('_p'), 'p', None))))
        amr = amr.assign(id_gen=np.where(amr.id.str.endswith('nonpara'), amr.id.str[:-8],
                                         np.where(amr.id.str.endswith('para'), amr.id.str[:-5],
                                                  np.where(amr.id.str.endswith('_p'), amr.id.str[:-2], None))))
        amr = amr.pivot(index=['id_gen'], values=['amr'], columns='id_type').reset_index()
        amr = amr.droplevel(level=0, axis=1)
        amr = amr.rename(columns={'': 'id'})
        amr_para = amr.loc[:, ['id', 'p', 'para']].rename(columns={'p': 'amr_p', 'para': 'amr_h'})
        amr_nonpara = amr.loc[:, ['id', 'p', 'nonpara']].rename(columns={'p': 'amr_p', 'nonpara': 'amr_h'})
        amr_nonpara['id'] = amr_nonpara['id'] + "_nonpara"
        amr_para['id'] = amr_para['id'] + "_para"

        amr_pivoted = pd.concat([amr_nonpara, amr_para])
        df = df.merge(amr_pivoted, how='inner', on='id')
        df['premise'] = df['input_json'].apply(lambda x: extract_value(x, 'premise'))
        df['hypothesis'] = df['input_json'].apply(lambda x: extract_value(x, 'hypothesis'))
        # Save column 'premise' and 'hypothesis' to two .txt files, where is each line is a sentence, name the file according to {dataset}_{column}.txt
        df['premise'].to_csv(data_dir / 'txt_files/ldc_dev_premise.txt', index=False, header=False)
        df['hypothesis'].to_csv(data_dir / 'txt_files/ldc_dev_hypothesis.txt', index=False, header=False)
        # Save columns 'amr_p' and 'amr_h' to two .txt files, where is each line is a AMR, name the file according to {dataset}_{column}.txt
        df['amr_p'].to_csv(data_dir / f'txt_files/{dataset}_amr_p.txt', index=False, header=False)
        df['amr_h'].to_csv(data_dir / f'txt_files/{dataset}_amr_h.txt', index=False, header=False)
    elif dataset in ['slang_gold', 'slang']:
        gold = pd.read_csv(data_dir/'classifier_inputs/ldc_slang_hand.csv')
        gold = gold[['id', 'true_premise_amr', 'hand_hypothesis_amr']]
        df['premise'] = df['input_json'].apply(lambda x: extract_value2(x, 'premise'))
        df['hypothesis'] = df['input_json'].apply(lambda x: extract_value2(x, 'hypothesis'))
        amr_og = amr.loc[amr.id.str.endswith('og')]
        amr_og['id_m'] = amr_og.id.str[:-3]
        amr_og = amr_og.loc[:, ['id_m', 'amr']].rename(columns={'amr': 'amr_h'})
        df['id_m'] = df.id.str[:13]
        amr = amr.rename(columns={'amr': 'amr_p'})
        df = df.merge(amr, how='inner', on='id').merge(amr_og, how='inner', on='id_m')
        df = df.merge(gold, how='inner', on='id')
        if 'gold' in dataset:
            df = df.drop(columns=['amr_p', 'amr_h'])
        else:
            df = df.drop(columns=['true_premise_amr', 'hand_hypothesis_amr'])
        # Save column 'premise' and 'hypothesis' to two .txt files, where is each line is a sentence, name the file according to {dataset}_{column}.txt
        df['premise'].to_csv(data_dir / f'txt_files/{dataset}_premise.txt', index=False, header=False)
        df['hypothesis'].to_csv(data_dir / f'txt_files/{dataset}_hypothesis.txt', index=False, header=False)
        # Save columns 'amr_p' and 'amr_h' to two .txt files, where is each line is a AMR, name the file according to {dataset}_{column}.txt
        if dataset in ['slang']:
            df['amr_p'].to_csv(data_dir / f'txt_files/{dataset}_amr_p.txt', index=False, header=False)
            df['amr_h'].to_csv(data_dir / f'txt_files/{dataset}_amr_h.txt', index=False, header=False)
        elif dataset in ['slang_gold']:
            df['true_premise_amr'].to_csv(data_dir / f'txt_files/{dataset}_amr_p.txt', index=False, header=False)
            df['hand_hypothesis_amr'].to_csv(data_dir / f'txt_files/{dataset}_amr_h.txt', index=False, header=False)


    if test_only:
        if dataset in ['paws', 'logic',  'django']:
            df = df.loc[df['id'].str.contains('test')]
        elif dataset in ['newstest']:
            df = df.loc[df['id'].str.contains('newstest16')]
        elif dataset in ['pubmed']:
            tmp = pd.read_csv(data_dir/"final_results/final_results_pubmed_corrected.csv")
            test_ids = tmp.id.values
            df = df[df['id'].isin(test_ids)]

    return df


def main():
    #loop through all datasets on process_data
    datasets = ['paws', 'django', 'logic', 'spider', 'entity_recog', 'entity_recog_gold', 'newstest', 'pubmed', 'ldc_dev', 'slang', 'slang_gold']
    # datasets = ['slang', 'slang_gold']
    # for dataset in datasets:
    #     print(dataset)
    #     df = process_data(data_file, amr_file, dataset, test_only = True)

    # loop through all .txt files in data_fir/'txt_files' and run the command to get perplexity
    for data_file in os.listdir(data_dir/'txt_files'):
        if data_file.endswith('.txt'):
            print(data_file)
            shell(f"python {root_dir.parent}/CausalLLMs/perplexity_calculator.py {data_dir}/txt_files/{data_file} --out-file {data_dir}/txt_files/{data_file[:-4]}_perplexity.json")


if __name__ == "__main__":
    main()
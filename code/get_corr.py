import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from pathlib import Path
import os
import json

root_dir = Path(__file__).parent.parent.resolve()
data_dir = root_dir / "data"
feature_dir = data_dir / "featured"
good_dir = data_dir / "good"
good_help_dir = data_dir / "good_with_help"
good_clean_dir = data_dir / "good_clean"
google_dir = r"~/Google Drive/My Drive/Zhijing&Yuen/amr_codes/data"
google_pred_dir = r"~/Google Drive/My Drive/Zhijing&Yuen/amr_codes/data/predictions"

def clean(df, save_path=None):
    if 'premise' in df.columns and 'hypothesis' in df.columns:
        for col in df.columns:
            if col.endswith('_pre') and col[:-4] + '_avg' not in df.columns:
                df[f"{col[:-4] + '_avg'}"] = (df[col] + df[f"{col[:-4] + '_hyp'}"]) / 2

        for col in df.columns:
            if col.endswith('_pre') or col.endswith('_hyp'):
                df = df.drop(columns=[col], axis=1)

    elif "en" in df.columns and 'de' in df.columns:
        for col in df.columns:
            if col.endswith('_de') and col[:-3] + '_avg' not in df.columns:
                df[f"{col[:-3] + '_avg'}"] = (df[col] + df[f"{col[:-4] + '_de'}"]) / 2

        for col in df.columns:
            if col.endswith('_de') or col.endswith('_en'):
                df = df.drop(columns=[col], axis=1)

    df.to_csv(save_path, index=False)







def get_correlation(df, dataset, target = 'helpfulness', save = True):
    for col in df.columns:
        if 'f1' in col or 'pred' in col or 'bleu' in col or "_pre" in col or "_hyp" in col or \
                "id_x" in col or "id_y" in col or "_en" in col or "_de" in col or 'label_' in col or 'invalid_' in col:
            df = df.drop(columns=[col], axis=1)
        elif "ground_truth" in col and target != 'ground_truth':
            df = df.drop(columns=[col], axis=1)
        elif "did_llm_failed" in col and target != 'did_llm_failed':
            df = df.drop(columns=[col], axis=1)
        elif "helpfulness" in col and target != 'helpfulness':
            df = df.drop(columns=[col], axis=1)



    if not target in df.columns:
        print(f"Target {target} not in {dataset} columns")
        return
    if target == 'helpfulness':
        file_name = f"{dataset}_helpfulness.csv"
    elif target == 'did_llm_failed':
        file_name = f"{dataset}_llm_fail.csv"
    elif target == 'bleu' and 'bleu' in df.columns:
        file_name = f"{dataset}_bleu.csv"
    elif target == 'ground_truth':
        file_name = f"{dataset}_repower.csv"

    corr_matrix = df.corr(numeric_only=True)
    if target in corr_matrix.index:
        corr_matrix = corr_matrix[[target]]
    else:
        print(f"Target {target} not in corr_matrix of {dataset}")
        return
    if save:
        corr_matrix.to_csv(data_dir / f"correlations/{file_name}", index=True)
    return corr_matrix


def combine_dataset(target = "helpfulenss"):
    if target == 'helpfulness':
        file_name = target
    elif target == 'did_llm_failed':
        file_name = 'llm_fail'
    elif target == 'bleu':
        file_name = 'bleu'
    elif target == 'ground_truth':
        file_name = 'repower'
    df = pd.DataFrame(columns=['feature'])
    for file in os.listdir(data_dir / "correlations"):
        if file.endswith(f"{file_name}.csv"):
            this_df = pd.read_csv(data_dir / f"correlations/{file}")
            this_df.columns = ['feature', file.replace(f"_{file_name}.csv", "")]
            this_df['feature'] = this_df['feature'].str.replace("_avg", "")
            df = df.merge(this_df, on = 'feature', how='outer')

    df.to_csv(data_dir / f"correlations/{target}_combined.csv", index=False)


if __name__ == "__main__":
    # for file in os.listdir(good_help_dir):
    #     if file.endswith(".csv") and not file.endswith("avg.csv"):
    #         df = pd.read_csv(good_help_dir / file)
    #         dataset = file.replace("_features.csv","").replace("_features_parser.csv","").replace("_features_true.csv","")
    #         clean(df, good_clean_dir / file)

    for file in os.listdir(good_clean_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(good_clean_dir / file)
            if 'string_len' not in df.columns:
                if 'premise' in df.columns and 'hypothesis' in df.columns:
                    df['string_len'] = (df['premise'].str.len() + df['hypothesis'].str.len()) / 2
                    df['num_word'] = (df['premise'].str.split().str.len() + df['hypothesis'].str.split().str.len()) / 2
                elif 'en' in df.columns and 'de' in df.columns:
                    df['string_len'] = (df['en'].str.len() + df['de'].str.len()) / 2
                    df['num_word'] = (df['en'].str.split().str.len() + df['de'].str.split().str.len()) / 2
                elif 'source_article' in df.columns:
                    df['string_len'] = df['source_article'].str.len()
                    df['num_word'] = df['source_article'].str.split().str.len()
                elif 'question' in df.columns:
                    df['string_len'] = df['question'].str.len()
                    df['num_word'] = df['question'].str.split().str.len()
                elif 'text' in df.columns:
                    df['string_len'] = df['text'].str.len()
                    df['num_word'] = df['text'].str.split().str.len()
                elif 'sentence' in df.columns:
                    df['string_len'] = df['sentence'].str.len()
                    df['num_word'] = df['sentence'].str.split().str.len()
                elif 'nl' in df.columns:
                    df['string_len'] = df['nl'].str.len()
                    df['num_word'] = df['nl'].str.split().str.len()

            df.to_csv(good_clean_dir / file, index=False)

            dataset = file.replace("_features.csv","").replace("_features_parser.csv","").replace("_features_true.csv","")
            get_correlation(df, dataset, target='helpfulness', save=True)
            get_correlation(df, dataset, target='did_llm_failed', save=True)
            get_correlation(df, dataset, target='bleu', save=True)
            get_correlation(df, dataset, target='ground_truth', save=True)

    for target in ['helpfulness', 'did_llm_failed', 'bleu', 'ground_truth']:
        print("Combining", target)
        combine_dataset(target)






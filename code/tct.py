import os
import sys
DC_HOME_DIR = "text_characterization_toolkit"
import text_characterization_toolkit
from text_characterization_toolkit.text_characterization.analysis import (
    show_pairwise_metric_correlations,
    PredictFromCharacteristicsAnalysis,
)
from text_characterization_toolkit.text_characterization.utils import load_text_metrics
import pandas as pd
# Execute this to get wide display
from IPython.display import display, HTML
import json
from pathlib import Path
import subprocess
import ast
from efficiency.function import shell

class TCT:
    def __init__(self, input_file, tct_tool_dir = '../../text_characterization_toolkit'):
        self.input_file = input_file
        self.tct_tool_dir = tct_tool_dir

    def convert_csv_to_jsonl(self, csv_file):
        df = pd.read_csv(csv_file)
        df = df.rename(columns={'id': 'id_temp'})
        df['id'] = list(range(len(df)))
        # Convert the DataFrame back to a list of dictionaries
        if 'premise' in df.columns:
            df = df[['id','premise','hypothesis']]
        elif 'en' in df.columns:
            df = df[['id','en']]
        elif 'text_detok' in df.columns:
            df = df[['id', 'text_detok']]
        elif 'text' in df.columns:
            df = df[['id','text']]
        elif 'sentence' in df.columns:
            df = df[['id','sentence']]
        elif 'source_article' in df.columns:
            df = df[['id','source_article']]
        elif 'nl' in df.columns:
            df = df[['id', 'nl']]
        elif 'question' in df.columns:
            df = df[['id', 'question']]

        for col in df.columns:
            if col not in ['id']:
                df[col] = df[col].astype(str)
        json_data = df.to_dict(orient="records")
        # json_file = csv_file.replace(".csv", ".jsonl")
        json_file = csv_file.with_suffix('.jsonl')
        # Write each dictionary in the list as a separate line in the output JSONL file
        with open(json_file, "w") as output_file:
            for item in json_data:
                json_line = json.dumps(item)
                output_file.write(json_line + "\n")


    def process_dataframe(self, metrics_df):
        # Check if multiple rows share the same index
        # if metrics_df.id.duplicated().any():
        if metrics_df.index.duplicated().any():

            # Pivot the dataframe
            # pivot_df = metrics_df.pivot_table(index='id', columns='text_key',
            #                                   values=metrics_df.columns.drop(['id', 'text_key']))

            pivot_df = metrics_df.pivot_table(index=metrics_df.index, columns='text_key',
                                              values=metrics_df.columns.drop('text_key'))

            # Rename the columns
            pivot_df.columns = [f"{col[0]}_{col[1][:3]}" for col in pivot_df.columns]
            # Reset the index
            pivot_df.reset_index(inplace=True)
            return pivot_df
        else:
            return metrics_df


    def get_tct(self):
        self.convert_csv_to_jsonl(self.input_file)
        jsonl_file = self.input_file.replace(".csv", ".jsonl")
        jsonl_file = self.input_file.with_suffix('.jsonl')
        tsv_output = self.input_file.with_suffix(".tsv")
        print(tsv_output)
        # Run the command using subprocess
        cmd = f"python \"{self.tct_tool_dir}/tools/compute.py\" -i {jsonl_file} -o {tsv_output}"
        stdout,stderr = shell(cmd)
        subprocess.run(
            ["python", f'{self.tct_tool_dir}/tools/compute.py', "-i", str(jsonl_file), "-o",
             str(tsv_output)])

        metrics_df = load_text_metrics(tsv_output)
        metrics_df.reset_index(inplace=True)
        pivot_df = self.process_dataframe(metrics_df)
        cols_to_drop = [col for col in pivot_df.columns if pivot_df[col].nunique() == 1]
        pivot_df = pivot_df.drop(cols_to_drop, axis=1)
        # merge back with feature_dir / file
        merged_df = pd.read_csv(self.input_file)
        merged_df = pivot_df.merge(merged_df, left_on='id', right_index=True)
        merged_df['id'] = merged_df['id_y']
        merged_df = merged_df.drop(['id_temp'], axis=1)
        return merged_df



def main():
    root_dir = Path(__file__).parent.parent.resolve()
    data_dir = root_dir / "data"
    feature_dir = data_dir / "featured"
    tct_dir = data_dir / "tct_outputs"
    parent_dir = os.path.dirname(root_dir)
    tct_tool_dir = os.path.join(parent_dir, DC_HOME_DIR)
    # tct = TCT(input_file, save_path, tct_tool_dir)
    # if file.endswith(".csv"):
    # if file == 'spider_text_features.csv' or file == 'django_text_features.csv':
    # print('Now processing: ', file)
    # with_tct = tct.get_tct(file)
    # with_tct.to_csv(tct_dir / file, index=False)
if __name__ == '__main__':
    main()

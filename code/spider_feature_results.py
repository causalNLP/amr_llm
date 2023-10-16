import pandas as pd
import os
from efficiency.function import set_seed
from pathlib import Path
import math
import matplotlib.pyplot as plt
import seaborn as sns


set_seed(0)
root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
out_dir = data_dir / "outputs"
spider_dir = out_dir /"spider_files"
parent_dir = os.path.dirname(root_dir)
# Initialize an empty list to store the dataframes

results = pd.read_csv(spider_dir/"gpt-4-0613/final_results_all_gpt-4-0613.csv")
features = pd.read_csv(data_dir/"featured/spider_features.csv")

df = results.merge(features, on='id', how = 'inner')
df['helpfulness'] = df['exact_match_pred_amr']-df['exact_match_pred']
df['did_llm_failed'] = df['exact_match_pred'].apply(lambda x: 1 if x==0 else 0)

df['helpfulness'].hist()
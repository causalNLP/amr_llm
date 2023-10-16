import pandas as pd
import os
from efficiency.function import set_seed
from pathlib import Path
import pdb
import math
import matplotlib.pyplot as plt
import seaborn as sns
import re


set_seed(0)
root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
out_dir = data_dir / "outputs"
spider_dir = out_dir /"spider_files"
parent_dir = os.path.dirname(root_dir)


all_features = pd.read_csv(data_dir/'all_data_features.csv')
amr_only_perp = pd.read_csv(data_dir/'amr_only_with_perplexity.csv')
amr_prompt_perp = pd.read_csv(data_dir/'prompts_amr_with_perplexity.csv')
direct_prompt_perp = pd.read_csv(data_dir/'prompts_direct_with_perplexity.csv')
pubmed_direct_prompt_perp = pd.read_csv(data_dir/'prompts_pubmed_direct_with_perplexity.csv')
pubmed_amr_prompt_perp = pd.read_csv(data_dir/'prompts_pubmed_amr_with_perplexity.csv')


amr_prompt_perp = pd.concat([amr_prompt_perp,pubmed_amr_prompt_perp])
amr_prompt_perp = amr_prompt_perp[~amr_prompt_perp['id'].str.contains('django')]


direct_prompt_perp = pd.concat([direct_prompt_perp,pubmed_direct_prompt_perp])
direct_prompt_perp = direct_prompt_perp[~direct_prompt_perp['id'].str.contains('django')]


merged_df = direct_prompt_perp[['id','perplexity']].merge(amr_prompt_perp[['id','perplexity']], on='id', suffixes=('_direct_prompt', '_amr_prompt'))
merged_df = merged_df.merge(amr_only_perp[['id','perplexity']], on='id')

merged_df.columns = ["id", "direct_prompt_perplexity", "amr_prompt_perplexity", "amr_perplexity"]

all_features = pd.merge(all_features, merged_df, on='id', how='left')

all_features.to_csv(data_dir/'all_data_features.csv',index = False)
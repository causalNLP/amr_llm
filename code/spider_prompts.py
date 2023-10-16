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

df = pd.read_csv(spider_dir/'gpt-4-0613/requests_spider_all.csv')

df = df.drop_duplicates(subset='id', keep='first')
#save column 'raw_prompt' to a txt file, separate by '\n'
# replace more than one "\s" with one " "

with open(data_dir/'amr_prompt.txt','w') as f:
    f.write('\n'.join(df['raw_prompt_amr'].astype(str).apply(
        lambda x: re.sub(' +', ' ', x.replace("\n", " ").replace("\t", " "))).tolist()))
with open(data_dir/'prompts_direct_spider.txt','w') as f:
    f.write('\n'.join(df['raw_prompt_direct'].astype(str).apply(
        lambda x: re.sub(' +', ' ', x.replace("\n", " ").replace("\t", " "))).tolist()))
with open(data_dir/'amr_only_spider.txt','w') as f:
    f.write('\n'.join(df['amr'].astype(str).apply(
        lambda x: re.sub(' +', ' ', x.replace("\n", " ").replace("\t", " "))).tolist()))



df[['id','amr','raw_prompt_amr','raw_prompt_direct']].to_csv(data_dir/'prompts_spider.csv',index=False)

print(df[['id','amr','raw_prompt_amr','raw_prompt_direct']].shape)
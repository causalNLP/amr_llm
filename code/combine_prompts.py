import pandas as pd
import os
from efficiency.function import set_seed
from pathlib import Path
import pdb
import math
import matplotlib.pyplot as plt
import seaborn as sns


set_seed(0)
root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
out_dir = data_dir / "outputs"
parent_dir = os.path.dirname(root_dir)

dataset = ['django','logic','paws','pubmed45','newstest']

df_direct = pd.DataFrame()
df_amr = pd.DataFrame()
# loop all files in through out_dir/'gpt-4-0613'
for filename in os.listdir(out_dir/'text-davinci-003'):
    if filename.endswith('.csv') and filename.replace("requests_amr_","") .replace("requests_direct_","").replace(".csv","") in dataset:
        filepath = os.path.join(out_dir / 'text-davinci-003', filename)
        df = pd.read_csv(filepath)
        if 'direct' in filename:
            df_direct = pd.concat([df_direct,df])
        else:
            df_amr = pd.concat([df_amr,df])

df_direct[['id','raw_prompt','pred']].to_csv(data_dir/'prompts_direct.csv',index=False)
df_amr[['id','raw_prompt','pred']].to_csv(data_dir/'prompts_amr.csv',index=False)

#save column 'raw_prompt' to a txt file, separate by '\n'
with open(data_dir/'prompts_direct.txt','w') as f:
    f.write('\n'.join(df_direct['raw_prompt'].astype(str).apply(lambda x: x.replace("\n"," ")).tolist()))
with open(data_dir/'prompts_amr.txt','w') as f:
    f.write('\n'.join(df_amr['raw_prompt'].astype(str).apply(lambda x: x.replace("\n"," ")).tolist()))

print(df_direct[['id','raw_prompt','pred']].shape)
print(df_amr[['id','raw_prompt','pred']].shape)
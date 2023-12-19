import pandas as pd
import os
from pathlib import Path


root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
out_dir = data_dir / "outputs"
parent_dir = os.path.dirname(root_dir)

model_version_dict = {
    'gpt4': "gpt-4-0613",
    'gpt3.5': "gpt-3.5-turbo-0613",
    'gpt3.043': "text-davinci-003",
    'gpt3.042': "text-davinci-002",
    'gpt3.041': "text-davinci-001",
}
# In out_dir, for each subdirectory whose name is in modle version dict.values, find the files with pattern "nottest" in their name
# For each file found, concatenate (using pandas) them to the file with the same name except _nottest is replaced by ''

for model_version in model_version_dict.values():
    if model_version in ["gpt-4-0613",'gpt-3.5-turbo-0613']:
        continue
    model_dir = out_dir / model_version
    for filename in os.listdir(model_dir):
        if "nottest" in filename and 'newstest' in filename:
            df = pd.read_csv(model_dir / filename)
            df_withtest = pd.read_csv(model_dir / filename.replace("_nottest", ""))
            if df_withtest.shape[0] > 3001:
                continue
            df = pd.concat([df, df_withtest])
            df.to_csv(model_dir / filename.replace("_nottest", ""), index=False)

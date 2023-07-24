import os
import json
from efficiency.nlp import Chatbot
from efficiency.log import fread, show_var, fwrite
from efficiency.function import random_sample, set_seed
import math
import random
import numpy as np
import re
import ipdb
from tqdm import tqdm
import pandas as pd
import itertools
from pathlib import Path
from sklearn.utils import shuffle
random.seed(0)
set_seed(0)



# model_version = ['gpt4', 'gpt3.5', 'gpt3.04', 'gpt3.043', 'gpt3.042', 'gpt3.041'][1]
model_version = 'gpt-3.5-turbo-0613'
# df = pd.read_csv("/content/drive/MyDrive/Zhijing&Yuen/gpt4_paws.csv")



# Enlarge paws to 16k

# os.chdir('drive/MyDrive/Zhijing&Yuen/amr_codes')
root_dir = Path(__file__).parent.parent.resolve()
data_dir = root_dir / "data"
current_dir = Path(__file__).parent.resolve()

def extract_mwes():
    # model_version = 'gpt4'
    model_version = 'gpt-3.5-turbo-0613'
    # save_path = google_dir
    save_path = data_dir / 'outputs'
    chat = Chatbot(model_version=model_version, max_tokens=30,
                   output_file=f'{save_path}/.cache_ldc_{model_version}_responses.csv',
                   system_prompt="You are a linguist of English and native English speaker.",
                   openai_key_alias='OPENAI_API_KEY',
                   )

    template_slang = "Please evaluate the following sentence for the presence of slang expressions. " \
                   "A slang expression is a phrase or expression that is in the online slang dictionaries and has meaning" \
                   "that is very different from its literal form. For instance, 'raining cats and dogs' " \
                   "is an slang, while 'middle school' is not. Although 'middle school' is a compound phrase, " \
                   "it does not carry a meaning beyond its literal interpretation. Here is the sentence for your analysis: {premise} " \
                   "Please format your response as follows:\n" \
                   "'{{Yes or No}},{{slangs}}.'\n" \
                   "If there's no slang used, just answer 'No'. If there are multiple slang expressions, please separate them with a semicolon (';').\n" \
                   "Remember, the idioms we are interested in are those that, when taken literally, would have a completely different semantic meaning."


    write_out_file = f"{data_dir}/{model_version}_ldc_slang.csv"
    read_in_file = f"{data_dir}/ldc_amr_clean.csv"

    slang_counts = 0
    df = pd.read_csv(read_in_file)
    df = shuffle(df)
    # num_processed = start_index
    dict_to_save = []
    for index, row in tqdm(df.iterrows()):
        print("Now processing ", index, row['id'])
        pred = chat.ask(template_slang.format(**{"premise": row['text_detok']}))
        norm = int(pred.split(",")[0].lower() == 'yes')
        if norm:
            slang_counts += 1
            print("There are ", slang_counts, " slangs collected.")
            slangs = pred.split(",")[1] if (len(pred.split(",")) > 1) else ''
            new_dict = {'id': row['id'],
                        # 'num_processed' : num_processed,
                        'text': row['text_detok'],
                        'true_amr': row['true_amr'],
                        "raw_pred": pred,
                        "contain_slangs": norm,
                        'slang_used': slangs}
            dict_to_save.append(new_dict)
            if slang_counts > 500:
                df = pd.DataFrame(dict_to_save)
                df.to_csv(write_out_file, index=False)
                break

        df = pd.DataFrame(dict_to_save)
        df.to_csv(write_out_file, index=False)


if __name__ == "__main__":
    extract_mwes()
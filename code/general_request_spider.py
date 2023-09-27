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
from efficiency.function import random_sample
from efficiency.nlp import Chatbot
from pathlib import Path
import tiktoken
from tqdm import tqdm
from efficiency.function import random_sample
from sklearn.utils import shuffle

np.random.seed(0)
random.seed(0)
set_seed(0)
root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
out_dir = data_dir / "outputs"
parent_dir = os.path.dirname(root_dir)
google_pred_dir = root_dir / "data/predictions"

prompts_dict = {
    "spider": {
        "system_prompt": """You are a language model designed to generate SQL queries based on natural language questions. Given a question, you need to generate the corresponding SQL query that retrieves the requested information from a database.""",
        "single_prompt": """Write an SQL query that retrieves the requested information based on the given natural language question. Remember to use proper SQL syntax and consider any necessary table joins or conditions.\nQuestion:{sentence_1}\nQuery:""",
        # "amr_prompt": """Write an SQL query that retrieves the requested information based on the given natural language question and its abstract meaning representation (AMR). Remember to use proper SQL syntax and consider any necessary table joins or conditions.\nQuestion:{sentence_1}\nAMR:\n{amr_1}\nQuery:""",
        "amr_prompt":"""\n#\n### For your reference, here is the abstract meaning representation (AMR) of the query:\n{amr}."""
    }
}


def clean_amr(amr):
    if not isinstance(amr, str):
        amr = str(amr)
    amr = re.sub("~e.\d+", "", amr)
    amr = re.sub("~\d+", "", amr)
    return amr



def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Request to openai models for amr project')
    parser.add_argument('-org', type=str,
                      default=["OPENAI_ZhijingPersonal_ID", "OPENAI_youdunn_ID", "OPENAI_JaiyiNLP_ID", "OPENAI_ORG_ID", ][-1],
                      help='put the ``')
    parser.add_argument('--data_file', type=str, default=data_dir / "final_results/final_results_spider_corrected.csv", help='the csv file')
    parser.add_argument('--dataset', type=str, default='spider', help='the dataset name')
    parser.add_argument('--model_version', type=str, default="gpt-3.5-turbo-16k-0613", help='which model to use')
    parser.add_argument('--amr_cot', type=bool, default=True, help='whether to use amr or not')
    args = parser.parse_args()
    return args


def main(file_path, dataset, amr_cot, model_version, num_samples, org_id = "OPENAI_ORG_ID"):
    df = pd.read_csv(file_path, sep=None, engine='python')
    if amr_cot:
        output_file = data_dir/f"outputs/{model_version}/requests_amr_{dataset}a.csv"
    else:
        output_file = data_dir/f"outputs/{model_version}/requests_direct_{dataset}.csv"
    ## setup chat
    save_path = data_dir / 'outputs'/ model_version
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    system_prompt = prompts_dict[dataset]['system_prompt']
    max_tokens = 300
    chat = Chatbot(model_version=model_version, max_tokens=max_tokens,
                      output_file=f'{save_path}/.cache_{model_version}_responses.csv',
                      system_prompt = system_prompt, openai_key_alias='OPENAI_API_KEY',
                        openai_org_alias=org_id
                      )


    # df = random_sample(df,df.shape[0])
    df = shuffle(df)
    df = df.reset_index(drop=True)


    ## requests
    #
    # which_part = all_orgs.index(org_id)
    # num_orgs = len(all_orgs)

    df['response'] = ''
    asked = 0
    for i, d in tqdm(df[:num_samples].iterrows(), total = num_samples, desc = "Requesting"):
        if amr_cot:
            prompt = df.loc[i, 'schema'] + prompts_dict[dataset]['amr_prompt'].format(amr=clean_amr(df.loc[i, 'amr']))
        else:
            prompt = df.loc[i, 'schema']
            
        df.loc[i, 'response'] = chat.ask(prompt)

        asked += 1

        # if i == 0:
            # print("Check system prompt: ", system_prompt)
            # print("Check system prompt used correctly ")
            # chat.ask("Who are you, python general_request_chatbot.py --model_version text-davinci-002 --dataset paws --org_id OPENAI_youdunn_ID and what's you task?", system_prompt=system_prompt, max_tokens=100)
        if i % 50 == 0:
            df.to_csv(output_file, index=False)

    print(output_file)
    df.to_csv(output_file, index=False)
    print(f'Save to {output_file}')



def get_args():
    parser = argparse.ArgumentParser(description='Request to openai models for amr project')
    parser.add_argument('-org_id', type=str,
                      default=["OPENAI_ZhijingPersonal_ID", "OPENAI_youdunn_ID", "OPENAI_JaiyiNLP_ID", "OPENAI_ORG_ID", ][-1],
                      help='put the ``')
    parser.add_argument('--data_file', type=str, default=data_dir/"final_results/final_results_spider_corrected.csv", help='the csv file')
    parser.add_argument('--dataset', type=str, default='spider', help='the dataset name')
    parser.add_argument('--model_version', type=str, default="text-davinci-001", help='which model to use')
    parser.add_argument('--amr_cot', type=bool, default=False, help='whether to use amr or not')
    parser.add_argument('--num_samples', type=int, default=100, help='how many samples to request')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    set_seed(0)
    args = get_args()
    model_version_dict = {
        'gpt4': "gpt-4-0613",
        'gpt3.5': "gpt-3.5-turbo-0613",
        'gpt3.043': "text-davinci-003",
        'gpt3.042': "text-davinci-002",
        'gpt3.041': "text-davinci-001",
    }

    #Samples 100 amrcot for paws
    main(args.data_file,args.dataset, args.amr_cot, args.model_version, args.num_samples)
    # main(data_file, amr_file, 'entity_recog_gold', True, 'text-davinci-001')



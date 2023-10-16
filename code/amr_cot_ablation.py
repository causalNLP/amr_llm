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
# google_amr_data_dir = r"~/Google Drive/My Drive/Zhijing&Yuen/amr_codes/data/"
# google_pred_dir = r"~/Google Drive/My Drive/Zhijing&Yuen/amr_codes/data/predictions"

prompts_dict = {
    "paws": {
        "system_prompt": """You are an NLP assistant whose purpose is to perform Paraphrase Identification. The goal of Paraphrase Identification is to determine whether a pair of sentences have the same meaning.""",
        "single_prompt": """Paraphrase Detection: Determine if the following two sentences are exact paraphrases (rewritten versions with the same meaning) of each other.\nSentence 1:{sentence_1}\nSentence 2:{sentence_2}\nAnswer [Yes/No] and then provide a brief explanation of why you think the sentences are paraphrases or not.\nParaphrase:""",
        "amr_prompt": """Paraphrase Detection: You are given two sentences and the abstract meaning representation (AMR) of each.\nSentence 1:{sentence_1}\nAMR 1:\n{amr_1}\nSentence 2:{sentence_2}\nAMR 2:\n{amr_2}\nExplain what are the commonalities and differences between the two AMRs. Then determine if the two sentences are exact paraphrases (rewritten versions with the same meaning) of each other and provide a brief explanation of why you think the sentences are paraphrases or not. Use the following format: Answer: [Yes/No]""",
    },
    "logic": {
        "system_prompt": """You are an expert in logic whose purpose is to determine the type of logical fallacy present in a text. The categories are: 1) Faulty Generalization\n2) False Causality\n3) Circular Claim\n4) Ad Populum\n5) Ad Hominem\n6) Deductive Fallacy\n7) Appeal to Emotion\n8) False Dilemma\n9) Equivocation\n10) Fallacy of Extension\n11) Fallacy of Relevance\n12) Fallacy of Credibility\n13) Intentional Fallacy.""",
        "single_prompt": """Please classify the following text into one of the logical fallacies: \nText:{sentence_1}\nWhich is the fallacy type present in the text?""",
        "amr_prompt": """You are given a text and its AMR.\nText:{sentence_1}\nAMR:\n{amr_1}\nBased on the text and its AMR please classify it into one of the logical fallacies. Which is the fallacy type present in the text?""",
    },
    "newstest": {
        "system_prompt": """You are an NLP assistant expert in machine translation from English to German.""",
        "single_prompt": """Please translate the following text from English to German.\nText: {sentence_1}\nTranslation:""",
        "amr_prompt": """You are given a text and its abstract meaning representation (AMR).\nText: {sentence_1}\nAMR:\n{amr_1}\nPlease translate the text from English to German. You can refer to the provided AMR if it helps you in creating the translation.\nTranslation:""",
    },
    "django": {
        "system_prompt": """You are an NLP assistant expert in translating natural language instructions to python code.""",
        "single_prompt": """Please generate python code instructions from the corresponding natural language descriptions. Exclude comments.\nDescription:{sentence_1}\nCode:""",
        "amr_prompt": """Please generate python code instructions from the corresponding natural language descriptions and its associated abstract meaning representation (AMR). Exclude comments.\nDescription:{sentence_1}\nAMR:\n{amr_1}\nCode:""",
    },
    "spider": {
        "system_prompt": """You are a language model designed to generate SQL queries based on natural language questions. Given a question, you need to generate the corresponding SQL query that retrieves the requested information from a database.""",
        "single_prompt": """Write an SQL query that retrieves the requested information based on the given natural language question. Remember to use proper SQL syntax and consider any necessary table joins or conditions.\nQuestion:{sentence_1}\nQuery:""",
        "amr_prompt": """Write an SQL query that retrieves the requested information based on the given natural language question and its abstract meaning representation (AMR). Remember to use proper SQL syntax and consider any necessary table joins or conditions.\nQuestion:{sentence_1}\nAMR:\n{amr_1}\nQuery:""",
    },
    "pubmed": {
        "system_prompt": "You are a medical professional expert.",
        "single_prompt": """This question aims to assess your proficiency in validating relationships between different entities in biomedical text. You will be presented with a sentence from an article and asked to determine whether the interaction between the entities mentioned in the sentence is valid or not. You should respond with a single digit, either "0" if the interaction is invalid, "1" if it is valid, or "2" if swapping the positions of any two entities would make the interaction valid. Please note that you are required to provide only one of these three responses.\nText: {sentence_1}\nInteraction: {interaction}""",
        "amr_prompt": """This question aims to assess your proficiency in validating relationships between different entities in biomedical text. You will be presented with a sentence from an article and its abstract meaning representation (AMR) and asked to determine whether the interaction between the entities mentioned in the sentence is valid or not. You should respond with a single digit, either "0" if the interaction is invalid, "1" if it is valid, or "2" if swapping the positions of any two entities would make the interaction valid. Please note that you are required to provide only one of these three responses.\nText: {sentence_1}\nAMR:\n{amr_1}\nInteraction: {interaction}""",
    },
    "entity_recog": {
        "system_prompt": """You are an NLP assistant whose purpose is to perform named entity recognition (NER).""",
        "single_prompt": """The following is a named entity recognition task. Please extract all the named entities of the following types from the given sentence.
TYPE="CARDINAL": Numerals that do not fall under another type, e.g., “one”, “ten”
TYPE="DATE": Absolute or relative dates or periods. E.g., “the summer of 2005”, “recent years”
TYPE="EVENT": Named hurricanes, battles, wars, sports events, etc. E.g., “Olympiad games”
TYPE="FAC": Buildings, airports, highways, bridges, etc. E.g., “Disney”, “the North Pole”
TYPE="GPE": Countries, cities, states. E.g., “Hong Kong”, “Putian”
TYPE="LAW": Named documents made into laws. E.g., “Chapter 11 of the federal Bankruptcy Code”
TYPE="LOC": Non-GPE locations, mountain ranges, bodies of water. E.g., “Mai Po Marshes”, “Asia”
TYPE="MONEY": Monetary values, including unit. E.g., “$ 1.3 million”, “more than $ 500 million”
TYPE="NORP": Nationalities or religious or political groups. E.g., “Chinese”, “Buddhism”
TYPE="ORDINAL": E.g., "first", "second", etc.
TYPE="ORG": Companies, agencies, institutions, etc. E.g., “Eighth Route Army”, “the Chinese Communist Party”
TYPE="PERCENT": Percentage, including "%". E.g., “25 %”
TYPE="PERSON": People, including fictional. E.g., “Zhu De”, “Saddam Hussein”
TYPE="PRODUCT":  Objects, vehicles, foods, etc. (Not services.) E.g., “iPhone”, “Coke Cola”
TYPE="QUANTITY": Measurements, as of weight or distance. E.g., “23 sq. km”
TYPE="TIME": Times smaller than a day. E.g., “homecoming night”
Sentence: {sentence_1}\nUse json format for the response where each key is an entity type.""",
        "amr_prompt": """The following is a named entity recognition task. Please extract all the named entities of the following types from the given sentence and its abstract meaning representation (AMR).
TYPE="CARDINAL": Numerals that do not fall under another type, e.g., “one”, “ten”
TYPE="DATE": Absolute or relative dates or periods. E.g., “the summer of 2005”, “recent years”
TYPE="EVENT": Named hurricanes, battles, wars, sports events, etc. E.g., “Olympiad games”
TYPE="FAC": Buildings, airports, highways, bridges, etc. E.g., “Disney”, “the North Pole”
TYPE="GPE": Countries, cities, states. E.g., “Hong Kong”, “Putian”
TYPE="LAW": Named documents made into laws. E.g., “Chapter 11 of the federal Bankruptcy Code”
TYPE="LOC": Non-GPE locations, mountain ranges, bodies of water. E.g., “Mai Po Marshes”, “Asia”
TYPE="MONEY": Monetary values, including unit. E.g., “$ 1.3 million”, “more than $ 500 million”
TYPE="NORP": Nationalities or religious or political groups. E.g., “Chinese”, “Buddhism”
TYPE="ORDINAL": E.g., "first", "second", etc.
TYPE="ORG": Companies, agencies, institutions, etc. E.g., “Eighth Route Army”, “the Chinese Communist Party”
TYPE="PERCENT": Percentage, including "%". E.g., “25 %”
TYPE="PERSON": People, including fictional. E.g., “Zhu De”, “Saddam Hussein”
TYPE="PRODUCT":  Objects, vehicles, foods, etc. (Not services.) E.g., “iPhone”, “Coke Cola”
TYPE="QUANTITY": Measurements, as of weight or distance. E.g., “23 sq. km”
TYPE="TIME": Times smaller than a day. E.g., “homecoming night”
Sentence: {sentence_1}\nAMR:\n{amr_1}\nUse json format for the response where each key is an entity type.""",
    },
}
prompts_dict['ldc_dev'] = prompts_dict['paws']
prompts_dict['slang'] = prompts_dict['paws']
prompts_dict['slang_gold'] = prompts_dict['paws']
prompts_dict['entity_recog_gold'] = prompts_dict['entity_recog']


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
    if 'gold' in dataset:
        df = df.loc[df.id.str.contains(dataset.replace('_gold', ''))]
        amr = amr.loc[amr.id.str.contains(dataset.replace('_gold', ''))]
    else:
        df = df.loc[df.id.str.contains(dataset)]
        amr = amr.loc[amr.id.str.contains(dataset)]
    df = df.loc[:, ['id', 'input_json', 'ground_truth']]
    if dataset in ['paws']:
        df = process_2_clauses(df, amr)
    elif dataset in ['django']:
        df['text'] = df['input_json'].apply(lambda x: extract_value(x, 'nl'))
        df = df.merge(amr, how='inner', on='id')
        df = df.loc[:, ['id', 'ground_truth', 'text', 'amr']].drop_duplicates()
    elif dataset in ['logic']:
        df['text'] = df['input_json'].apply(lambda x: extract_value(x, 'source_article'))
        df = df.merge(amr, how='inner', on='id')
    elif dataset in ['spider']:
        df['text'] = df['input_json'].apply(lambda x: extract_value(x, 'question'))
        df = df.merge(amr, how='inner', on='id')
    elif dataset in ['entity_recog_gold']:
        gold = pd.read_csv('../data/ldc_ner_features_true.csv')
        gold = gold[['id', 'true_amr']]
        gold['true_amr'] = gold['true_amr'].apply(lambda x: clean_amr(x))
        df = df.merge(gold, how='inner', on='id')
        df['text'] = df['input_json'].apply(lambda x: extract_value(x, 'text'))
        df = df.merge(amr, how='inner', on='id')
    elif dataset in ['entity_recog']:
        df['text'] = df['input_json'].apply(lambda x: extract_value(x, 'text'))
        df = df.merge(amr, how='inner', on='id')
    elif dataset in ['newstest']:
        df['text'] = df['input_json'].apply(lambda x: extract_value(x, 'en'))
        df['ground_truth'] = df['input_json'].apply(lambda x: extract_value(x, 'de'))
        amr['id'] = amr['id'].str[:-3]
        df = df.merge(amr, how='inner', on='id')
    elif dataset in ['pubmed']:
        df['text'] = df['input_json'].apply(lambda x: extract_value(x, 'sentence'))
        df['interaction'] = df['input_json'].apply(lambda x: extract_value(x, 'interaction'))
        df = df.merge(amr, how='inner', on='id')
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
    elif dataset in ['slang_gold', 'slang']:
        gold = pd.read_csv('../data/classifier_inputs/ldc_slang_hand.csv')
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

    if test_only:
        if dataset in ['paws', 'logic', 'pubmed', 'django']:
            df = df.loc[df['id'].str.contains('test')]
        elif dataset in ['newstest']:
            df = df.loc[df['id'].str.contains('newstest2016')]

    return df


def clean_amr(amr):
    if not isinstance(amr, str):
        amr = str(amr)
    amr = re.sub("~e.\d+", "", amr)
    return amr



def process_cut(df, cut_cols = ['true_amr'], keep_list=[0.1, 0.3, 0.4, 0.6, 0.7]):
    if isinstance(keep_list, float) or isinstance(keep_list, int):
        keep_list = [keep_list]
    for cut_col in cut_cols:
        df[f'{cut_col}_keep_ratio'] = [keep_list] * len(df)

        # Use explode to expand the DataFrame
        df = df.explode(f'{cut_col}_keep_ratio')
        df = df.reset_index(drop=True)
        for i, row in df.iterrows():
            if row[f'{cut_col}_keep_ratio'] == 0:
                df.loc[i, cut_col] = ''
            elif row[f'{cut_col}_keep_ratio'] == 1:
                continue
            else:
                if 'amr' in cut_col:
                    df.loc[i, cut_col] = cut_amr(row[cut_col], row[f'{cut_col}_keep_ratio'])
                else:
                    df.loc[i, cut_col] = cut_text(row[cut_col], row[f'{cut_col}_keep_ratio'])
    return df



def process_response(df, dataset, amr_cot):
    if dataset in ['paws', 'ldc_dev', 'slang', 'slang_gold']:
        df['response_final'] = df['response']
        if amr_cot:
            df['response_final'] = df['response_final'].fillna('')
            df['response_final'] = df['response_final'].apply(
                lambda x: x if x is None else x.split('Answer:')[1] if 'Answer:' in x else x)
            df['response_final'] = df['response_final'].str.strip()
            # df['response_final'] = df['response_final'].str.split('Answer:').str[1]
            # df['response_final'] = df['response_final'].str.strip()
            # df['response_final'] = df['response_final'].fillna('')
        df = df.assign(pred=np.where(df.response_final.str.lower().str.startswith('yes'), 1,
                                     np.where(df.response_final.str.lower().str.startswith('no'), 0, np.NaN)))
    elif dataset in ['newstest', 'django', 'spider', 'entity_recog', 'pubmed', 'entity_recog_gold']:
        df['pred'] = df['response']
    elif dataset in ['logic']:
        df['pred'] = ''
        df = df.assign(
            pred=np.where(df.response.str.lower().str.contains('faulty generalization'), 'Faulty Generalization',
                          df.pred))
        df = df.assign(
            pred=np.where(df.response.str.lower().str.contains('false causality'), 'False Causality', df.pred))
        df = df.assign(
            pred=np.where(df.response.str.lower().str.contains('circular claim'), 'Circular Reasoning', df.pred))
        df = df.assign(pred=np.where(df.response.str.lower().str.contains('ad populum'), 'Ad Populum', df.pred))
        df = df.assign(pred=np.where(df.response.str.lower().str.contains('ad hominem'), 'Ad Hominem', df.pred))
        df = df.assign(
            pred=np.where(df.response.str.lower().str.contains('deductive fallacy'), 'fallacy of logic', df.pred))
        df = df.assign(
            pred=np.where(df.response.str.lower().str.contains('appeal to emotion'), 'Appeal to Emotion', df.pred))
        df = df.assign(pred=np.where(df.response.str.lower().str.contains('false dilemma'), 'False Dilemma', df.pred))
        df = df.assign(pred=np.where(df.response.str.lower().str.contains('equivocation'), 'Equivocation', df.pred))
        df = df.assign(
            pred=np.where(df.response.str.lower().str.contains('fallacy of extension'), 'Fallacy of Extension',
                          df.pred))
        df = df.assign(
            pred=np.where(df.response.str.lower().str.contains('fallacy of relevance'), 'Fallacy of Relevance',
                          df.pred))
        df = df.assign(
            pred=np.where(df.response.str.lower().str.contains('fallacy of credibility'), 'Fallacy of Credibility',
                          df.pred))
        df = df.assign(
            pred=np.where(df.response.str.lower().str.contains('intentional fallacy'), 'Intentional', df.pred))
        df['pred'] = df['pred'].str.lower()
    return df


def simple_evaluation(df, test_set_pattern):
    df = df.loc[df.pred != '']
    df = df.loc[~df.pred.isna()]
    df.ground_truth = df.ground_truth.apply(int)
    df_test = df.loc[df.id.str.contains(test_set_pattern)]
    print("Data points: ", df_test.shape[0])
    print("f1-score positive class:",
          classification_report(df_test.ground_truth, df_test.pred, output_dict=True)['1']['f1-score'])
    print(classification_report(df.ground_truth, df.pred))
    return df


def simple_evaluation_str(df, test_set_pattern):
    df = df.loc[df.pred != '']
    df = df.loc[~df.pred.isna()]
    df_test = df.loc[df.id.str.contains(test_set_pattern)]
    print("Data points: ", df.shape[0])
    print("f1-score micro /accuracy:", classification_report(df.ground_truth, df.pred, output_dict=True)['accuracy'])
    print(classification_report(df.ground_truth, df.pred))
    return df


def bleu_evaluation(df, test_set_pattern):
    df['bleu'] = 0
    df = df.loc[df.pred != '']
    df = df.loc[~df.pred.isna()]
    for i, d in df.iterrows():
        score = 0
        answer = d['pred'].replace("\n", "\\n")
        score = list_bleu([[d['ground_truth']]], [answer])
        df.at[i, 'bleu'] = score
    df_test = df.loc[df.id.str.contains(test_set_pattern)]
    print("Data points: ", df_test.shape[0])
    print("Avg BLEU:", df_test.bleu.mean())
    return df


def extract_entities(text):
    entity_pattern = r'<ENAMEX TYPE="([^"]*)">([^<]*)</ENAMEX>'
    entities = re.findall(entity_pattern, text)
    entity_dict = {}
    for entity_type, entity_value in entities:
        if entity_type in entity_dict:
            entity_dict[entity_type].append(entity_value)
        else:
            entity_dict[entity_type] = [entity_value]
    return entity_dict


def ner_evaluation(df, test_set_pattern):
    # gt=pd.read_csv("./data/classifier_inputs/ldc_entity_recog_to_classifier.csv")
    gt = pd.read_csv("../data/classifier_inputs/ldc_ner_to_classifier.csv")
    gt['labels'] = gt['input_json'].apply(lambda x: extract_value(x, 'tok_labeled'))
    gt = gt.loc[:, ['id', 'labels']]
    df = df.merge(gt, on='id')
    df = df.loc[~df.pred.isna()]
    df = df.loc[df.pred != '']
    # Apply the function to the dataframe column
    df['entities'] = df['labels'].apply(extract_entities)
    df['pred'] = df['pred'].apply(ast.literal_eval)
    df['f1'] = 0

    for i, row in df.iterrows():
        ground_truth = row['entities']
        prediction = row['pred']

        ground_truth_set = set(
            (entity_type, entity_value) for entity_type, entity_values in ground_truth.items() for entity_value in
            entity_values)
        prediction_set = set(
            (entity_type, entity_value) for entity_type, entity_values in prediction.items() for entity_value in
            entity_values)
        if len(prediction_set) == 0 or len(ground_truth_set) == 0:
            # print(prediction_set)
            # print(ground_truth_set)
            f1 = 0
            df.at[i, 'f1'] = f1
            continue
        true_positives = len(ground_truth_set.intersection(prediction_set))
        false_positives = len(prediction_set - ground_truth_set)
        false_negatives = len(ground_truth_set - prediction_set)

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        df.at[i, 'f1'] = f1
    df_test = df.loc[df.id.str.contains(test_set_pattern)]
    print("Data points: ", df_test.shape[0])
    print("Avg F1:", df_test.f1.mean())
    return df





def cut_amr(amr_str, keep=1):
    if not isinstance(amr_str, str):
        amr_str = str(amr_str)

    amr_str = clean_amr(amr_str)
    amr_str = amr_str.replace('\n', '$$newline$$')
    amr_str = amr_str.replace('\t', '$$tab$$')

    amr_tokens = amr_str.split(' ')
    num_seq = [i for i, _ in enumerate(amr_tokens)]
    set_seed(0)
    num_seq = random_sample(num_seq, None)

    num_seq_to_keep = num_seq[:int(len(num_seq) * keep)]
    amr_tokens = [amr for num, amr in enumerate(amr_tokens) if num in num_seq_to_keep]
    amr_new_str = ' '.join(amr_tokens)
    amr_new_str = amr_new_str.replace('$$newline$$', "\n").replace('$$tab$$', "\t")

    # Verify that the ratio is correct
    # if keep == 0.2:
    #     print("orignal amr:", amr_str)
    #     print("cut amr:", amr_new_str)
    #     print(len(amr_new_str.split(' '))/len(amr_str.split(' ')))
    return amr_new_str


def cut_text(text, keep=1):
    if not isinstance(text, str):
        text = str(text)

    text = clean_amr(text)
    text = text.replace('\n', '$$newline$$')
    text = text.replace('\t', '$$tab$$')

    text_tokens = text.split(' ')
    num_seq = [i for i, _ in enumerate(text_tokens)]
    set_seed(0)
    num_seq = random_sample(num_seq, None)

    num_seq_to_keep = num_seq[:int(len(num_seq) * keep)]
    text_tokens = [amr for num, amr in enumerate(text_tokens) if num in num_seq_to_keep]
    text_new_str = ' '.join(text_tokens)
    text_new_str = text_new_str.replace('$$newline$$', "\n").replace('$$tab$$', "\t")

    # Verify that the ratio is correct
    # if keep == 0.2:
    #     print("orignal amr:", amr_str)
    #     print("cut amr:", amr_new_str)
    #     print(len(amr_new_str.split(' '))/len(amr_str.split(' ')))
    return text_new_str





def main(dataset, output_file, cut_col, keep_ratio, amr_cot = True):
    data_file = data_dir / "classifier_inputs/updated_data_input - classifier_input.csv"
    amr_file = data_dir / "corrected_amrs.csv"
    model_version = "gpt-3.5-turbo-0613"
    save_path = out_dir / model_version
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    system_prompt = prompts_dict[dataset]['system_prompt']
    # max_tokens = prompts_dict[dataset]['max_tokens']
    max_tokens = 800

    enc = tiktoken.encoding_for_model(model_version)
    chat = Chatbot(model_version=model_version, max_tokens=max_tokens,
                      output_file= f'{save_path}/.cache_{model_version}_responses_amr_ablation.csv',
                      system_prompt = system_prompt, openai_key_alias='OPENAI_API_KEY'
                      )
    chat.clear_dialog_history()
    if amr_cot:
        prompt = prompts_dict[dataset]['amr_prompt']
    else:
        prompt = prompts_dict[dataset]['single_prompt']


    df = process_data(data_file, amr_file, dataset)
    if cut_col == 'text':
        if 'premise' in dataset:
            cut_cols = ['premise', 'hypothesis']
        elif dataset in ['logic']:
            cut_cols = ['source_article']
        elif dataset in ['pubmed']:
            cut_cols = ['sentence']
        elif 'en' in df.columns:
            cut_cols = ['en']
        elif dataset in ['django']:
            cut_cols = ['nl']
        elif dataset in ['entity_recog', 'entity_recog_gold']:
            cut_cols = ['text']
        elif 'question' in df.columns:
            cut_cols = ['question']
    if cut_col == 'amr':
        cut_cols = []
        if 'true_premise_amr' in df.columns:
            cut_cols.append('true_premise_amr')
        if 'hand_hypothesis_amr' in df.columns:
            cut_cols.append('hand_hypothesis_amr')
        if 'premise_amr' in df.columns:
            cut_cols.append('premise_amr')
        if 'hypothesis_amr' in df.columns:
            cut_cols.append('hypothesis_amr')
        if 'amr_p' in df.columns:
            cut_cols.append('amr_p')
        if 'amr_h' in df.columns:
            cut_cols.append('amr_h')
        if 'amr' in df.columns:
            cut_cols.append('amr')



    df = process_cut(df, cut_cols = cut_cols,keep_list= keep_ratio)
    # df = random_sample(df,df.shape[0])



    df['response'] = ''
    asked = 0
    df = df.reset_index(drop=True)
    for i, d in tqdm(df.iterrows(), total = df.shape[0]):
        if 'pred' in df.columns and d['pred'] in [0,1]:
            continue
        # if i % num_orgs != which_part:
        #     continue
        if dataset in ['slang_gold']:
            m1 = prompt.format(sentence_1=d['premise'], amr_1=d['true_premise_amr'], sentence_2=d['hypothesis'],
                               amr_2=d['hand_hypothesis_amr'])
        elif dataset in ['entity_recog']:
            m1 = prompt.format(sentence_1=d['text'], amr_1=d['amr'])
        elif dataset in ['entity_recog_gold']:
            m1 = prompt.format(sentence_1=d['text'], amr_1=d['true_amr'])
        elif dataset in ['paws']:
            m1 = prompt.format(sentence_1=d['premise'], amr_1=d['amr_p'].replace(" " * 4, "\t"),
                               sentence_2=d['hypothesis'], amr_2=d['amr_h'].replace(" " * 4, "\t"))
        elif dataset in ['ldc_dev', 'slang']:
            m1 = prompt.format(sentence_1=d['premise'], amr_1=d['amr_p'], sentence_2=d['hypothesis'], amr_2=d['amr_h'])
        elif dataset in ['newstest', 'logic', 'django', 'spider', 'entity_recog']:
            m1 = prompt.format(sentence_1=d['text'], amr_1=d['amr'])
        elif dataset in ['pubmed']:
            m1 = prompt.format(sentence_1=d['text'], amr_1=d['amr'], interaction=str(d['interaction']))
        df.at[i, 'raw_prompt'] = m1
        df.loc[i, 'response'] = chat.ask(m1,system_prompt=system_prompt)

        if i % 50 == 0:
            df.to_csv(output_file, index=False)

    df.to_csv(output_file, index=False)
    print(f'Save to {output_file}')

    df = process_response(df, dataset, amr_cot)
    df.to_csv(output_file, index=False)

def get_args():
    parser = argparse.ArgumentParser(description='Request to openai models for amr project')
    parser.add_argument('-org_id', type=str,
                      default=["OPENAI_ORG_ID", ][-1],
                      help='put the ``')
    parser.add_argument('--data_file', type=str, default=data_dir/"classifier_inputs/updated_data_input - classifier_input.csv", help='the csv file')
    parser.add_argument('--amr_file', type=str, default=data_dir/'corrected_amrs.csv',  help='the amr csv file')
    parser.add_argument('--dataset', type=str, default='logic', help='the dataset name')
    parser.add_argument('--model_version', type=str, default="text-davinci-001", help='which model to use')
    parser.add_argument('--amr_cot', type=bool, default=False, help='whether to use amr or not')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    set_seed(0)


    # args = get_args()
    parser = argparse.ArgumentParser(description='Request to openai models for amr project')

    parser.add_argument('--dataset', type=str, default='entity_recog', help='the dataset name')
    parser.add_argument('--cut_col', type=str, default='amr', help='which column to cut')
    parser.add_argument('--output_file', type=str, default = './test.csv', help='whether to use amr or not')
    parser.add_argument('--ratio', type=float, default=0.5, help='whether to use amr or not')
    args = parser.parse_args()


    model_version_dict = {
        'gpt4': "gpt-4-0613",
        # 'gpt3.5': "gpt-3.5-turbo-0613",
        'gpt3.043': "text-davinci-003",
        'gpt3.042': "text-davinci-002",
        'gpt3.041': "text-davinci-001",
    }

    print(args.cut_col)
    # main(args.dataset, args.output_file, cut_col = args.cut_col, keep_ratio = args.ratio)
    # main('entity_recog', data_dir/'ablations/text_ablation.csv', cut_col = 'text', keep_ratio = np.linspace(0.1, 1, 10))
    # main('entity_recog', data_dir/ 'ablations/text_ablation.csv', cut_col = 'text', keep_ratio = 1)
    main('entity_recog', data_dir / 'ablations/text_ablation_1_only.csv', cut_col='text', keep_ratio=1)


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
from pathlib import Path
import pdb
import math


np.random.seed(0)
random.seed(0)
set_seed(0)
root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
out_dir = data_dir / "outputs"
parent_dir = os.path.dirname(root_dir)
google_amr_data_dir = r"~/Google Drive/My Drive/Zhijing&Yuen/amr_codes/data/"
google_pred_dir = r"~/Google Drive/My Drive/Zhijing&Yuen/amr_codes/data/predictions"

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


def process_response(df, dataset, amr_cot):
    if dataset in ['paws', 'ldc_dev', 'slang', 'slang_gold']:
        df['response_final'] = df['response']
        if amr_cot:
            df['response_final'] = df['response_final'].fillna('')
            def helper(x):
                if 'Answer:' in x:
                    return x.split('Answer:')[1]
                elif 'are paraphrases of each other' in x or "\nYes" in x or "the answer is yes" in x.lower():
                    return'Yes'
                elif x.startswith("The sentences are paraphrases") or x.startswith("Yes"):
                    return'Yes'
                elif 'are not paraphrases of each other' in x or 'are not exact paraphrases' in x or '\nNo' in x or x.startswith("No") or "the answer is no" in x.lower():
                    return "No"
                elif '\n\n' in x:
                    return x.split('\n\n')[1]
                else:
                    return x

            df['response_final'] = df['response_final'].apply(helper)
            df['response_final'] = df['response_final'].str.strip()
            # df['response_final'] = df['response_final'].str.split('Answer:').str[1]
            # df['response_final'] = df['response_final'].str.strip()
            df['response_final'] = df['response_final'].fillna('')
        df = df.assign(pred=np.where(df.response_final.str.lower().str.startswith('yes'), 1,
                                     np.where(df.response_final.str.lower().str.startswith('no'), 0, np.NaN)))
    elif dataset in ['newstest', 'django', 'spider', 'entity_recog', 'pubmed', 'entity_recog_gold']:
        df['pred'] = df['response']
        df.loc[df['response'] == 'valid', 'pred'] = 1
        df.loc[df['response'] == 'invalid', 'pred'] = 0
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
    # elif dataset in ['logic']:
    #     # Initialize pred to 'None'
    #     df['pred'] = 'None'
    #
    #     # Then apply all your conditions
    #     conditions = [
    #         ('faulty generalization', 'Faulty Generalization'),
    #         ('false causality', 'False Causality'),
    #         ('circular claim', 'Circular Reasoning'),
    #         ('ad populum', 'Ad Populum'),
    #         ('ad hominem', 'Ad Hominem'),
    #         ('deductive fallacy', 'fallacy of logic'),
    #         ('appeal to emotion', 'Appeal to Emotion'),
    #         ('false dilemma', 'False Dilemma'),
    #         ('equivocation', 'Equivocation'),
    #         ('fallacy of extension', 'Fallacy of Extension'),
    #         ('fallacy of relevance', 'Fallacy of Relevance'),
    #         ('fallacy of credibility', 'Fallacy of Credibility'),
    #         ('intentional fallacy', 'Intentional')
    #     ]
    #
    #     for keyword, label in conditions:
    #         df['pred'] = np.where(df['response'].str.lower().str.contains(keyword), label, df['pred'])

    # Your return statement
    return df


def simple_evaluation(df, test_set_pattern):
    df = df.loc[df.pred != '']
    df = df.loc[~df.pred.isna()]

    df_test = df.loc[df.id.str.contains(test_set_pattern)]
    print("Data points: ", df_test.shape[0])
    print("f1-score positive class:",
          classification_report(df_test.ground_truth, df_test.pred, output_dict=True)['1']['f1-score'])
    print(classification_report(df.ground_truth, df.pred, digits=4))
    return df


def simple_evaluation_str(df, test_set_pattern):
    df = df.loc[df.pred != '']
    df = df.loc[~df.pred.isna()]

    def try_convert_to_int(x):
        try:
            return int(x)
        except ValueError:  # Handle specific exception
            return -1

    # df['pred'] = df['pred'].apply(try_convert_to_int)

    df_test = df.loc[df.id.str.contains(test_set_pattern)]
    print("Data points: ", df.shape[0])
    print("f1-score micro /accuracy:", classification_report(df.ground_truth, df.pred, output_dict=True)['accuracy'])
    print(classification_report(df.ground_truth, df.pred, digits = 4))
    return df


def bleu_evaluation(df, test_set_pattern):
    df['bleu'] = 0
    df = df.loc[df.pred != '']
    df = df.loc[~df.pred.isna()]
    def short_bleu(ref, hyp):
        if ref == hyp:
            return 1
        multiplier = int(math.ceil(4 / min(len(ref.split()), len(hyp.split()))))
        ref = f"{(ref + ' ') * multiplier}".strip()
        hyp = f"{(hyp + ' ') * multiplier}".strip()
        return list_bleu([[ref]], [hyp])

    for i, d in df.iterrows():
        score = 0

        answer = d['pred'].replace("```python\n", "")

        answer = answer.replace("\n```", "")
        answer = answer.replace("\n", "\\n")
        answer = answer.replace("\"", "")

        score = list_bleu([[d['ground_truth']]], [answer])
        if score== 0:
            score = short_bleu(d['ground_truth'], answer)
        df.at[i, 'pred'] = answer
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
    gt=pd.read_csv(data_dir/"classifier_inputs/ldc_ner_to_classifier.csv")
    gt['labels'] = gt['input_json'].apply(lambda x: extract_value(x, 'tok_labeled'))
    gt=gt.loc[:,['id','labels']]
    if 'labels' not in df.columns:
        df=df.merge(gt,on='id')
    # print(df)
    df=df.loc[~df.pred.isna()]
    df=df.loc[df.pred!='']
    # Apply the function to the dataframe column
    df['entities'] = df['labels'].apply(extract_entities)

    def safe_literal_eval(s):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s  # or return None, or some default value

    df['pred'] = df['pred'].apply(lambda x: safe_literal_eval(x))
    df['f1']=0
    for i, row in df.iterrows():
        ground_truth = row['entities']
        prediction = row['pred']

        ground_truth_set = set(
            (entity_type, entity_value) for entity_type, entity_values in ground_truth.items() for entity_value in
            entity_values)
        try:
            prediction_set = set(
                (entity_type, entity_value) for entity_type, entity_values in prediction.items() for entity_value in
                entity_values)
        except Exception as e:
            print(i ,prediction_set)
            prediction_set = set()

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







def main(file_path, dataset, amr_cot):
    print("Performance Test on " + file_path)
    df = pd.read_csv(file_path)
    df = process_response(df, dataset, amr_cot)
    if dataset in ['paws']:
        simple_evaluation(df, 'test')
    elif dataset in ['ldc_dev', 'slang', 'slang_gold']:
        simple_evaluation(df, dataset.replace('_gold', ''))
    elif dataset in ['logic']:
        simple_evaluation_str(df, "test")
    elif dataset in ['pubmed']:
        simple_evaluation_str(df, "test")
    elif dataset in ['newstest']:
        df = bleu_evaluation(df, 'newstest16')
    elif dataset in ['django']:
        df = bleu_evaluation(df, 'test')
    elif dataset in ['entity_recog', 'entity_recog_gold']:
        df = ner_evaluation(df, 'entity_recog')
    df.to_csv(file_path, index=False)



if __name__ == '__main__':
    set_seed(0)
    # parser = argparse.ArgumentParser(description='Request to openai models for amr project')
    # parser.add_argument('--file_path', type=str, help = 'the csv file to evaluate')
    # parser.add_argument('--dataset', type=str, default='logic', help='the dataset name')
    # parser.add_argument('--model_version', type=str, default="text-davinci-001", help='which model to use')
    # parser.add_argument('--amr_cot', action = 'store_true', default=False, help='whether to use amr or not')
    # args = parser.parse_args()
    # main(args.file_path, args.dataset)
    model_list = ['text-davinci-002', 'text-davinci-001', 'text-davinci-003','gpt-4-0613']
    # main(f"{out_dir}/gpt-4-0613/requests_direct_newstest.csv","newstest",False)
    for model, t in zip(model_list,['direct', 'amr']):
        main(f"{out_dir}/{model}/requests_{t}_django.csv", "django", False)
        main(f"{out_dir}/{model}/requests_{t}_logic.csv", "logic", False)
    # main(f"{out_dir}/gpt-3.5-turbo-0613/requests_amr_django.csv", "django", False)
    # main("/Users/chenyuen/Desktop/amr_llm/data/ablations/text_ablation_1_only.csv", "entity_recog", True)
    # for m in model_list:
    #     for file in os.listdir(out_dir/m):
    #         if file.endswith(".csv") and not file.startswith("._") and not file.startswith(".cache"):
    #             try:
    #                 if 'paws' in file or 'djando' in file or 'newstest' in file or 'spider' in file or 'logic' in file or 'pubmed' in file:
    #                     dataset = file.replace("requests_", "").replace(".csv", "").replace("direct_", "").replace("amr_", "")
    #                     if 'amr' in file:
    #                         main(os.path.join(out_dir / m, file), dataset,True)
    #                     else:
    #                         main(os.path.join(out_dir / m, file), dataset, False)
    #             except Exception as e:
    #                 print(e)

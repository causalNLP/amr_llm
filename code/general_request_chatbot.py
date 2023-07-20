from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
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


def process_data(file_path, file_path_amr, dataset):
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

    return df


def clean_amr(amr):
    if not isinstance(amr, str):
        amr = str(amr)
    amr = re.sub("~e.\d+", "", amr)
    return amr


def process_cut(df, keep_list=[0.1, 0.3, 0.4, 0.6, 0.7]):
    # rest of your code):
    df['amr_keep_ratio'] = [keep_list] * len(df)
    # Use explode to expand the DataFrame
    df = df.explode('amr_keep_ratio')
    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        if row['amr_keep_ratio'] == 0:
            df.loc[i, 'true_amr'] = ''
        elif row['amr_keep_ratio'] == 1:
            continue
        else:
            df.loc[i, 'true_amr'] = cut_amr(row['true_amr'], row['amr_keep_ratio'])
    return df


def process_response(df, dataset, amr_cot):
    if dataset in ['paws', 'ldc_dev', 'slang', 'slang_gold']:
        df['response_final'] = df['response']
        if amr_cot:
            df['response_final'] = df['response_final'].str.split('Answer:').str[1]
            df['response_final'] = df['response_final'].str.strip()
            df['response_final'] = df['response_final'].fillna('')
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


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Request to openai models for amr project')
    parser.add_argument('-org', type=str,
                      default=["OPENAI_ZhijingPersonal_ID", "OPENAI_youdunn_ID", "OPENAI_JaiyiNLP_ID", "OPENAI_ORG_ID", ][-1],
                      help='put the ``')
    parser.add_argument('--data_file', type=str, default="./updated_data_input - classifier_input.csv", help='the csv file')
    parser.add_argument('--amr_file', type=str, default='./corrected_amrs.csv',  help='the amr csv file')
    parser.add_argument('--dataset', type=str, default='logic', help='the dataset name')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo-16k-0613", help='which model to use')
    args = parser.parse_args()
    return args


def main(file_path, file_path_amr, dataset, amr_cot, model_version, org_id = "OPENAI_ORG_ID"):
    ## parameters
    # dataset='logic'
    # all_datasets=['newstest','paws','django','logic','spider','entity_recog','pubmed','ldc_dev']
    # amr_cot=True
    # file_path="./updated_data_input - classifier_input.csv"
    # file_path_amr="./corrected_amrs.csv"
    if amr_cot:
        output_file = "../data/outputs/requests_amr_" + dataset + "_effic_0720.csv"
    else:
        output_file = "../data/outputs/requests_direct_" + dataset + "_effic_0720.csv"

    ## setup chat
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613")
    save_path = data_dir / 'outputs'
    system_prompt = prompts_dict[dataset]['system_prompt']
    chat = Chatbot(model_version=model_version, max_tokens=2048,
                      output_file=f'{save_path}/.cache_{model_version}_responses.csv',
                      system_prompt = system_prompt, openai_key_alias='OPENAI_API_KEY',
                        openai_org_alias=org_id
                      )
    chat.clear_dialog_history()
    if amr_cot:
        prompt = prompts_dict[dataset]['amr_prompt']
    else:
        prompt = prompts_dict[dataset]['single_prompt']

    df = process_data(file_path, file_path_amr, dataset)
    # df=process_cut(df)

    # sys_prompt = ChatPromptTemplate.from_messages([
    #     SystemMessagePromptTemplate.from_template(
    #         system_prompt
    #     ),
    #     MessagesPlaceholder(variable_name="history"),
    #     HumanMessagePromptTemplate.from_template('{input}')
    # ])

    ## requests
    df['response'] = ''

    for i, d in df.iterrows():
        # memory = ConversationBufferMemory(return_messages=True)
        # conversation = ConversationChain(memory=memory, prompt=sys_prompt, llm=llm)

        if dataset in ['slang_gold']:
            m1 = prompt.format(sentence_1=d['premise'], amr_1=d['true_premise_amr'], sentence_2=d['hypothesis'],
                               amr_2=d['hand_hypothesis_amr'])
        elif dataset in ['entity_recog']:
            m1 = prompt.format(sentence_1=d['text'], amr_1=d['true_amr'])
        elif dataset in ['entity_recog_gold']:
            m1 = prompt.format(sentence_1=d['text'], amr_1=d['amr'])
        elif dataset in ['paws', 'ldc_dev', 'slang']:
            m1 = prompt.format(sentence_1=d['premise'], amr_1=d['amr_p'], sentence_2=d['hypothesis'], amr_2=d['amr_h'])
        elif dataset in ['newstest', 'logic', 'django', 'spider', 'entity_recog']:
            m1 = prompt.format(sentence_1=d['text'], amr_1=d['amr'])
        elif dataset in ['pubmed']:
            m1 = prompt.format(sentence_1=d['text'], amr_1=d['amr'], interaction=str(d['interaction']))
        df.at[i, 'raw_prompt'] = m1
        # if pd.isna(df.at[i, 'response']):
        #     print('Now querying: ', i)
        df.loc[i, 'response'] = chat.ask(m1,system_prompt=system_prompt)

        # history = memory.chat_memory.messages
        if i % 50 == 0:
            print(i)
            print(d['id'], "gt:", d['ground_truth'], "#### pred: ", df.at[i, 'response'])
            df.to_csv(output_file, index=False)

    # parse response and results
    # df = pd.read_csv(output_file)
    df = process_response(df, dataset, amr_cot)
    df.to_csv(output_file, index=False)

    print("Performance Test")
    if dataset in ['paws']:
        simple_evaluation(df, 'dev')
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

    df.to_csv(output_file, index=False)


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




if __name__ == '__main__':
    set_seed(0)
    args = get_args()
    print(args.org)
    data_file = "../data/classifier_inputs/ldc_slang_to_classifier.csv"
    amr_file = "../data/corrected_amrs.csv"
    amr_cot = False
    model_version = "gpt-3.5-turbo-16k-0613"
    for data_set in ['slang']:
        # main(args.data_file, args.amr_file,args.dataset,amr_cot)
        main(data_file, amr_file, data_set, amr_cot, model_version, args.org)
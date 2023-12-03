import json
import os
import random
from efficiency.function import set_seed
import pandas as pd
import re
import sys
from pathlib import Path
# np.random.seed(0)
# random.seed(0)
set_seed(0)
root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
out_dir = data_dir / "outputs"
parent_dir = os.path.dirname(root_dir)
google_pred_dir = root_dir / "data/predictions"
prompts_dict = {
    "paws": {
        "system_prompt": """You are an NLP assistant whose purpose is to perform Paraphrase Identification. The goal of Paraphrase Identification is to determine whether a pair of sentences have the same meaning.""",
        "single_prompt": """Paraphrase Detection: Determine if the following two sentences are exact paraphrases (rewritten versions with the same meaning) of each other.\nSentence 1:{sentence_1}\nSentence 2:{sentence_2}\nAnswer [Yes/No] and then provide a brief explanation of why you think the sentences are paraphrases or not.\nParaphrase:""",
        "amr_prompt": """Paraphrase Detection: You are given two sentences and the abstract meaning representation (AMR) of each.\nSentence 1:{sentence_1}\nAMR 1:\n{amr_1}\nSentence 2:{sentence_2}\nAMR 2:\n{amr_2}\nExplain what are the commonalities and differences between the two AMRs. Then determine if the two sentences are exact paraphrases (rewritten versions with the same meaning) of each other and provide a brief explanation of why you think the sentences are paraphrases or not. Use the following format: Answer: [Yes/No]""",
    },
    "logic": {
        "system_prompt": """You are an expert in logic whose purpose is to determine the type of logical fallacy present in a text. The categories are: 1) "Faulty Generalization"\n2) "False Causality"\n3) "Circular Claim"\n4) "Ad Populum"\n5) "Ad Hominem"\n6) "Deductive Fallacy"\n7) "Appeal to Emotion"\n8) "False Dilemma"\n9) "Equivocation"\n10) "Fallacy of Extension"\n11) "Fallacy of Relevance"\n12) "Fallacy of Credibility"\n13) "Intentional Fallacy".""",
        "single_prompt": """Please classify the following text into one of the logical fallacies: \nText:{sentence_1}\nWhich is the fallacy type present in the text?""",
        "amr_prompt": """You are given a text and its AMR.\nText:{sentence_1}\nAMR:\n{amr_1}\nBased on the text and its AMR please classify it into one of the logical fallacies. Which is the fallacy type present in the text? Please only chose a type from the given categories.""",
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
        "amr_prompt": """\n#\n### For your reference, here is the abstract meaning representation (AMR) of the query:\n{amr}."""
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



prompts_dict['paws']['single_prompt'] = """Paraphrase Detection: Determine if the following two sentences are exact paraphrases (rewritten versions with the same meaning) of each other.

Example 1:
{example1}
Answer: {ground_truth1}

Example 2:
{example2}
Answer: {ground_truth2}\n\nSentence 1:{sentence_1}\nSentence 2:{sentence_2}\nAnswer [Yes/No] and then provide a brief explanation of why you think the sentences are paraphrases or not.\nParaphrase:"""

prompts_dict['paws']['amr_prompt'] = """Paraphrase Detection: You are given two sentences and the abstract meaning representation (AMR) of each.

Example 1:
{example1}
Answer: {ground_truth1}

Example 2:
{example2}
Answer: {ground_truth2}

Sentence 1:{sentence_1}\nAMR 1:\n{amr_1}\nSentence 2:{sentence_2}\nAMR 2:\n{amr_2}\nExplain what are the commonalities and differences between the two AMRs. Then determine if the two sentences are exact paraphrases (rewritten versions with the same meaning) of each other and provide a brief explanation of why you think the sentences are paraphrases or not. Use the following format: Answer: [Yes/No]"""


prompts_dict['logic']['single_prompt'] = """Please classify the following text into one of the logical fallacies:

Example 1:
{example1}
Answer: {ground_truth1}

Example 2:
{example2}
Answer: {ground_truth2}

Text:{sentence_1}\nWhich is the fallacy type present in the text?"""

prompts_dict['logic']['amr_prompt'] = """Please classify the following text into one of the logical fallacies:

Example 1:
{example1}
Answer: {ground_truth1}

Example 2:
{example2}
Answer: {ground_truth2}

Text:{sentence_1}\nAMR:\n{amr_1}\nWhich is the fallacy type present in the text?"""


prompts_dict['newstest']['single_prompt'] = """Please translate the following text from English to German.

Example 1:
{example1}
{ground_truth1}

Example 2:
{example2}
{ground_truth2}

Text: {sentence_1}\nTranslation:"""

prompts_dict['newstest']['amr_prompt'] = """Please translate the following text from English to German.\nExample 1:\n{example1}\nExample 2:\n{example2}\n\nText: {sentence_1}\nAMR:\n{amr_1}\nTranslation:"""

prompts_dict['spider']['single_prompt'] ="""Example 1:
{example_1}
{ground_truth1}

Example 2:
{example_2}
{ground_truth2}

{schema}"""



prompts_dict['spider']['amr_prompt'] = """Example 1:
{example_1}
{ground_truth1}

Example 2:
{example_2}
{ground_truth2}

{schema} \n\n### For your reference, here is the abstract meaning representation (AMR) of the query:\n{amr}."""




prompts_dict['pubmed']['single_prompt'] = """This question aims to assess your proficiency in validating relationships between different entities in biomedical text. You will be presented with a sentence from an article and asked to determine whether the interaction between the entities mentioned in the sentence is valid or not. You should respond with a single digit, either "0" if the interaction is invalid, "1" if it is valid, or "2" if swapping the positions of any two entities would make the interaction valid. Please note that you are required to provide only one of these three responses.

Example 1:
{example1}
{ground_truth1}

Example 2:
{example2}
{ground_truth2}


Text: {sentence_1}\nInteraction: {interaction}"""

prompts_dict['pubmed']['amr_prompt'] = """This question aims to assess your proficiency in validating relationships between different entities in biomedical text. You will be presented with a sentence from an article and its abstract meaning representation (AMR) and asked to determine whether the interaction between the entities mentioned in the sentence is valid or not. You should respond with a single digit, either "0" if the interaction is invalid, "1" if it is valid, or "2" if swapping the positions of any two entities would make the interaction valid. Please note that you are required to provide only one of these three responses.

Example 1:
{example1}
{ground_truth1}

Example 2:
{example2}
{ground_truth2}

Text: {sentence_1}\nAMR:\n{amr_1}\nInteraction: {interaction}"""

prompts_dict['entity_recog']['single_prompt'] = """The following is a named entity recognition task. Please extract all the named entities of the following types from the given sentence.
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

Example 1:
{example1}
{ground_truth1}

Example 2:
{example2}
{ground_truth2}

Sentence: {sentence_1}
Use json format for the response where each key is an entity type."""


prompts_dict['entity_recog']['amr_prompt'] = """The following is a named entity recognition task. Please extract all the named entities of the following types from the given sentence and its abstract meaning representation (AMR).
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

Example 1:
{example1}
{ground_truth1}

Example 2:
{example2}
{ground_truth2}

Sentence: {sentence_1}\nAMR:\n{amr_1}\nUse json format for the response where each key is an entity type."""


prompts_dict['ldc_dev'] = prompts_dict['paws']
prompts_dict['asilm'] = prompts_dict['paws']
prompts_dict['slang'] = prompts_dict['paws']
prompts_dict['slang_gold'] = prompts_dict['paws']
prompts_dict['entity_recog_gold'] = prompts_dict['entity_recog']




def get_sampele(dataset, df, amr = False, n = 1):
    if dataset in ['spider']:
        input_file = out_dir / f"spider_files/gpt-4-0613/requests_spider_all.csv"
    else:
        input_file = data_dir / f"outputs_gpt4/requests_{'amr' if amr else ''}_{dataset}.csv"


    example_df = pd.read_csv(input_file)


    # Find two rows in df with different ground_truth
    different_ground_truth_rows = df[df['ground_truth'].diff().ne(0)].tail(2)

    if len(different_ground_truth_rows) < 2:
        raise ValueError("Not enough rows with different ground truths found.")

    # Extract IDs
    ids = different_ground_truth_rows['id'].tolist()

    # Choose rows from example_df with the same IDs
    sample_rows = example_df[example_df['id'].isin(ids)]

    # take the last 2 samples as 2-shot examples
    ground_truth1 = sample_rows[0]['ground_truth']
    ground_truth2 = sample_rows[1]['ground_truth']


    if dataset in ['spider']:
        example1 = sample_rows[0]['raw_prompt_amr' if amr else 'raw_prompt_direct']
        example2 = sample_rows[1]['raw_prompt_amr' if amr else 'raw_prompt_direct']
    else:
        example1 = sample_rows[0]['raw_prompt']
        example2 = sample_rows[1]['raw_prompt']

    # return a dict
    return {
        'example1': example1,
        'example2': example2,
        'ground_truth1': ground_truth1,
        'ground_truth2': ground_truth2,
    }

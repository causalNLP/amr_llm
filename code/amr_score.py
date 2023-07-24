from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import json
import os
import smatch
import pandas as pd
from pathlib import Path
import numpy as np
import re
import datasets
import ast
from tqdm import tqdm
from efficiency.function import  shell
from efficiency.log import fwrite, fread
import contextlib
import io

# from efficiency.log import fwrite, fread


root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
output_dir = data_dir / "outputs"
tct_out_dir = data_dir / "tct_outputs"
parent_dir = os.path.dirname(root_dir)
onto_dir = f'{parent_dir}/ontonotes-release-5.0'
model_dir = root_dir / "model"
feature_dir = data_dir / "featured"
sample_dir = data_dir / "samples"
google_dir = r"~/Google Drive/My Drive/Zhijing&Yuen/"
google_amr_data_dir = r"~/Google Drive/My Drive/Zhijing&Yuen/amr_codes/data/"
google_pred_dir = r"~/Google Drive/My Drive/Zhijing&Yuen/amr_codes/data/predictions"


def extract_amr(s):
    match = re.search(r'\. \((.*)\)', s)
    if match:
        return "(" + match.group(1) + ")"

    match = re.search(r'\: \((.*)\)', s)
    if match:
      return "(" + match.group(1) + ")"

    match = re.search(r'\" \((.*)\)', s)
    if match:
      return "(" + match.group(1) + ")"

    match = re.search(r' \((.*)\)', s)
    if match:
      return "(" + match.group(1) + ")"

    else:
      # print("No match in ", s)
      return None


def replace_sentence(s):
    if s is None:
        return None
    pattern = r"The abstract meaning representation.*as follows:"
    return re.sub(pattern, '', s).strip()

def parse_string(s):
    count_open = 0
    count_close = 0
    if s is None:
        return None
    try:
        s = s.replace("[", "(").replace("]", ")")
        s = re.sub("~\d+", "", s)
        s = re.sub("\t", " ", s)
        s = re.sub("\s+", " ", s)
    except Exception as e:
        return None

    for i in range(len(s)-1, -1, -1):
        if s[i] == ')':
            count_close += 1
        elif s[i] == '(':
            count_open += 1
        if count_open == count_close:
            return s[i:]
    return s




def parse_string_simple(s):
    def escape_slash_not_surrounded_by_spaces(text):
        return re.sub(r'(?<! )/(?! )', r'\/', text)
    s = escape_slash_not_surrounded_by_spaces(s)
    s = s.replace("[", "(").replace("]", ")")
    s = re.sub("~\d+", "", s)
    s = re.sub("\t", " ", s)
    s = re.sub("\s+", " ", s)
    return s

def balance_parentheses(s):
    if s is None:
        return None
    stack = []
    for i in range(len(s)):
        if s[i] == '(':
            stack.append(i)
        elif s[i] == ')':
            if stack:
                stack.pop()
            else:
                s = s[:i] + s[i+1:]
                return balance_parentheses(s)
    while stack:
        i = stack.pop()
        s = s[:i] + s[i+1:]
    return s


def premise_generator(premise_list):
    for premise in premise_list:
        yield premise.strip()

def hypothesis_generator(hypothesis_list):
    for hypothesis in hypothesis_list:
        yield hypothesis.strip()

def compute_smatch_for_pairs(premise_list, hypothesis_list):
    import warnings
    import smatch
    scores = []
    invalid_amr = []
    invalid_type = []
    invalid_count = 0
    for premise, hypothesis in tqdm(zip(premise_list, hypothesis_list)):
        if premise is None or hypothesis is None:
            scores.append(0)
            continue

        try:
            with warnings.catch_warnings(record=True) as w:
                smatch.match_triple_dict.clear()
                best_match_num, test_triple_num, gold_triple_num = smatch.get_amr_match(premise, hypothesis)
                precision, recall, score = smatch.compute_f(best_match_num, test_triple_num, gold_triple_num)
                score = min(score, 1)
                scores.append(score)
                smatch.match_triple_dict.clear()
                # Check if any warnings were captured
                if w:
                    # If so, append the warning message to invalid_type
                    # invalid_type.append(str(w[-1].message))
                    invalid_type.append(None)
                    invalid_amr.append(0)
                    invalid_count += 1
                else:
                    invalid_amr.append(0)
                    invalid_type.append(None)
        except Exception as e:
            invalid_amr.append(1)
            print('invalid amr', hypothesis)
            invalid_type.append(str(e))
            invalid_count += 1
            scores.append(None)  # or some other default score
    print('len(invalid_type)', len(invalid_type))
    return scores, invalid_count, invalid_amr, invalid_type



def instance_relation_match(amr1_input, amr2_input):
  if amr1_input is None or amr2_input is None:
    return 0,0

  try:
      amr1 = smatch.amr.AMR.parse_AMR_line(amr1_input)
      amr2 = smatch.amr.AMR.parse_AMR_line(amr2_input)
  except Exception as e:
      return 0,0
  if amr1 is None or amr2 is None:
    return 0,0
  instance_t, relation_t = amr1.get_triples2()
  instance_t2, relation_t2 = amr2.get_triples2()
  def quantify_similarity(list1, list2):
      matches = sum(1 for a, b in zip(list1, list2) if a == b)
      avg_length = (len(list1) + len(list2)) / 2
      return matches / avg_length

  instance_match = quantify_similarity(instance_t, instance_t2)
  relation_match = quantify_similarity(relation_t, relation_t2)

  return instance_match, relation_match








#################### AMR complexity features ####################
def amr_depth(amr):
    max_depth = 0
    current_depth = 0
    if not isinstance(amr, str):
        return 0
    for char in amr:
        if char == '(':
            current_depth += 1
            if current_depth > max_depth:
                max_depth = current_depth
        elif char == ')':
            if current_depth > 0:
                current_depth -= 1
    return max_depth

def amr_width(amr):
    # iterate through every "(", count the number of direct ":" not inside any other "("
    # or ask chatGPT, with two example amrs.
    max_width = 0
    current_depth = 0
    widths = [0]
    if not isinstance(amr, str):
        return 0
    for char in amr:
        if char == '(':
            current_depth += 1
            if current_depth >= len(widths):
                widths.append(0)
        elif char == ')':
            if current_depth > 0:
                current_depth -= 1
        elif char == ':':
            widths[current_depth] += 1
            max_width = max(max_width, widths[current_depth])

    return max_width

def unique_roles(amr):
    if not isinstance(amr, str):
        return 0
    role_pattern = re.compile(r':[a-zA-Z0-9_-]+')
    roles = set(role_pattern.findall(amr))
    return len(roles)


def amr_tokens(amr):
    if not isinstance(amr, str):
        return 0
    tokens = nltk.word_tokenize(amr)
    return len(tokens)



###### for paired amrs ######
def get_3_amr_features(df, amr_pred ='premise_amr', amr_gold='hypothesis_amr'):
  '''Given a df containing columns ['premise_amr','hypothesis_amr'],
  add three more columns to df ,['smatch_score','instance_match', 'relation_match']'''

  premise_list = df[amr_pred].tolist()
  # print(f"{len(premise_list)} amrs from premises")
  hypothesis_list = df[amr_gold].tolist()
  # print(f"{len(hypothesis_list)} amrs from hypotheses")

  df['smatch_score'], invalid_counts, df['invalid_amr'], df['invalid_type'] = compute_smatch_for_pairs(premise_list, hypothesis_list)
  df['instance_match'] = df.apply(lambda row: instance_relation_match(row[amr_pred], row[amr_gold])[0], axis=1)
  df['relation_match'] = df.apply(lambda row: instance_relation_match(row[amr_pred], row[amr_gold])[1], axis=1)
  print(f"{invalid_counts} invalid amrs", f"{len(premise_list)-invalid_counts} valid amrs")
  mean_score = df['smatch_score'].dropna().mean()
  print("Mean smatch score:", mean_score)
  return df



#### For single amrs ####
def get_amr_features_one_sent(df, amr_col ='amr', col_name_add = ''):
    func_list = [amr_depth, amr_width, unique_roles, amr_tokens]
    for func in func_list:
        df[f'{func.__name__}{col_name_add}'] = df[amr_col].apply(func)
    return df


def get_amr_features_two_sent(df, amr_col1 ='premise_amr', amr_col2 ='hypothesis_amr'):
    func_list = [amr_depth, amr_width, unique_roles, amr_tokens]
    for func in func_list:
        df[f'{func.__name__}_{amr_col1[:3]}'] = df.apply(lambda row: func(row[amr_col1]), axis=1)
        df[f'{func.__name__}_{amr_col2[:3]}'] = df.apply(lambda row: func(row[amr_col2]), axis=1)
        df[f'{func.__name__}_avg'] = (df[f'{func.__name__}_{amr_col1[:3]}'] + df[f'{func.__name__}_{amr_col2[:3]}'])/2
    return df


def amr_lbl():
    dat = pd.read_csv(data_dir / 'paws_amr_30.csv')
    hyp_file = 'tmp_hyp.txt'
    prem_file = 'tmp_prem.txt'
    data = dat[dat['sembleu'] == 'Auto_reweigh, max-gram is 2 new weight is (0.5, 0.5)']
    for index, row in tqdm(data.iterrows()):
        hyp = row['hypothesis_amr']
        prem = row['premise_amr']
        fwrite(hyp, hyp_file)
        fwrite(prem, prem_file)
        os.chdir(f'{parent_dir}/sembleu')
        cmd = f'{parent_dir}/sembleu/eval.sh {current_dir}/tmp_hyp.txt {current_dir}/tmp_prem.txt'
        stdout, stderr = shell(cmd)
        os.chdir(current_dir)
        stdout = stdout.split('\n')
        sembleu_idx = stdout.index('evaluating ...') + 1
        smatch = stdout[sembleu_idx]
        if 'Auto_reweigh, max-gram is 2 new weight is (0.5, 0.5)' in stdout[sembleu_idx]:
            smatch = stdout[sembleu_idx + 1]
        data.loc[index, 'sembleu'] = smatch
        if index % 100 == 0:
            print(f'index: {index}, smatch: {smatch}')
            data.to_csv(data_dir / 'paws_amr_30.csv', index=False)
    data.to_csv(data_dir / 'paws_amr_30.csv', index=False)



def scoring_parser_bleu():
    df = pd.read_csv(data_dir / 'sentence2amr.csv')

    df_ldc_test = df[df['id'].str.contains('ldc') & df['id'].str.contains('test')].copy()
    df_ldc_test.loc[:, 'amr'] = df_ldc_test['amr'].apply(extract_amr)
    df_ldc_test.loc[:, 'amr'] = df_ldc_test['amr'].apply(parse_string)

    df_ldc_test.to_csv(f'{data_dir}/ldc_test.csv', index=False)
    # save 'premise_amr'to one file and 'hypothesis_amr' to another, where datapoints are separated by an empty line
    with open(f'{data_dir}/ldc_gold_amr.txt', 'w') as f:
        for item in df_ldc_test['true_amr']:
            item = re.sub("~e.\d+", "", item)
            f.write("%s\n\n" % item)

    with open(f'{data_dir}/ldc_parsed_amr.txt', 'w') as f:
        for item in df_ldc_test['amr']:
            f.write("%s\n\n" % item)

    os.chdir(f'{parent_dir}/sembleu')
    cmd = f'{parent_dir}/sembleu/eval.sh {data_dir}/ldc_parsed_amr.txt {data_dir}/ldc_gold_amr.txt'
    stdout, stderr = shell(cmd)
    stdout = stdout.split('\n')
    sembleu_idx = stdout.index('evaluating ...') + 1
    sembleu = stdout[sembleu_idx]
    print(sembleu)
    os.chdir(current_dir)



def scoring_parser_smatch():
    df = pd.read_csv(data_dir / 'sentence2amr.csv')

    df_ldc_test = df[df['id'].str.contains('ldc') & df['id'].str.contains('test')].copy()
    df_ldc_test.loc[:, 'amr'] = df_ldc_test['amr'].apply(extract_amr)
    df_ldc_test.loc[:, 'amr'] = df_ldc_test['amr'].apply(parse_string)
    df_ldc_test.loc[:, 'true_amr'] = df_ldc_test['true_amr'].apply(lambda x: re.sub("~e.\d+", "", x))
    df_ldc_test = get_3_amr_features(df_ldc_test, amr_pred='amr', amr_gold='true_amr')
    df_ldc_test.to_csv(f'{data_dir}/ldc_test.csv', index=False)
    # save 'premise_amr'to one file and 'hypothesis_amr' to another, where datapoints are separated by an empty line


def scoring_gpt_smatch():
    df_ldc_test = pd.read_csv(f'{data_dir}/ldc_test.csv')
    # df = pd.read_csv(data_dir / 'outputs/gpt-3.5-turbo-0613_ldc_test_amrs_old.csv')
    # df = pd.read_csv(data_dir / 'outputs/gpt-3.5-turbo-0613_ldc_test_amrs_full.csv')
    # read_in = data_dir / 'outputs/gpt-3.5-turbo-0613_ldc_invalid_amrs_retry.csv'
    # read_in = data_dir / 'gpt-3.5-turbo-0613_ldc_test_amrs_old.csv'
    # read_in = output_dir / 'gpt-3.5-turbo-0613_ldc_amr_2.0.csv'
    read_in = output_dir / 'gpt4-0613_ldc_amr_2.0.csv'
    df = pd.read_csv(read_in)
    print(df.shape)
    for col in df.columns:
        if "Unnamed" in col:
            df = df.drop(col, axis=1)
    # df.loc[:, 'gpt_amr'] = df['gpt-3.5-turbo-0613_amr'].apply(lambda x: x.replace("( (", "((").replace(") )", "))"))

    # find the true amr in ldc_test by id
    if 'true_amr' not in df.columns:
        df = df.merge(df_ldc_test[['id', 'true_amr']], on='id', how='left')
    # df = df[~df['gpt4_amr'].isna()]
    # df = df[~df['gpt-3.5-turbo-0613_amr'].isna()]
    df = df[~df['gpt4-0613_amr'].str.contains('None')]
    df.loc[:, 'gpt_amr'] = df['gpt4-0613_amr'].apply(parse_string_simple)
    # df.loc[:, 'gpt_amr'] = df['gpt_amr'].apply(lambda x: x.replace("unknown",":name (n / name :op1 'Unknown')"))
    # df.loc[:, 'gpt_amr'] = df['gpt_amr'].apply(extract_amr)
    # df.loc[:, 'gpt3_amr'] = df['gpt3_amr'].apply(balance_parentheses)
    df.loc[:, 'true_amr'] = df['true_amr'].apply(lambda x: re.sub("~e.\d+", "", x))
    df = get_3_amr_features(df, amr_pred='gpt_amr', amr_gold='true_amr')
    print(df['smatch_score'].mean())

    # df.to_csv(read_in, index=False)

    # to_inspect_path = data_dir/'gpt4-0613_invalid_amrs.csv'
    # to_inspect = df[df['invalid_amr'] == 1]
    # to_inspect.to_csv(to_inspect_path, index=False)


def get_amr_features(input_file):
    df = pd.read_csv(input_file)
    df.loc[:, 'amr'] = df['amr'].apply(extract_amr)
    df.loc[:, 'amr'] = df['amr'].apply(parse_string)
    df.loc[:, 'true_amr'] = df['true_amr'].apply(lambda x: re.sub("~e.\d+", "", x))
    df = get_3_amr_features(df, amr_pred='amr', amr_gold='true_amr')
    return df

def main():
    # amr_lbl()
    # scoring_parser_smatch()d
    # scoring_gpt_smatch()
    # df_ldc_test = pd.read_csv(f'{data_dir}/ldc_test.csv')
    # for file in os.listdir(tct_out_dir):
    #     if file.endswith('.csv'):
    #         df = pd.read_csv(tct_out_dir/file)
    #         df = get_amr_features_one_sent(df)
    #         df.to_csv(tct_out_dir/file, index=False)


    # ldc_slang = pd.read_csv(feature_dir / 'ldc_slang_text_features.csv')
    # ldc_slang_reseult = pd.read_csv(f'{google_pred_dir}/final_results_paraphrase_slang_gold.csv')
    # ldc_slang['premise_amr'] = ldc_slang_reseult['amr_p']
    # ldc_slang['hypothesis_amr'] = ldc_slang_reseult['amr_h']
    # ldc_slang = get_amr_features_one_sent(ldc_slang, amr_col = 'premise_amr', col_name_add = '_pre')
    # ldc_slang = get_amr_features_one_sent(ldc_slang, amr_col = 'hypothesis_amr', col_name_add = '_hyp')
    # ldc_slang = get_3_amr_features(ldc_slang, amr_pred='premise_amr', amr_gold='hypothesis_amr')
    # ldc_slang.to_csv(feature_dir / 'ldc_slang_features_parser.csv', index=False)

    # spider = pd.read_csv(tct_out_dir/'spider_text_features.csv')
    # spider = get_amr_features_one_sent(spider)
    # spider.to_csv(tct_out_dir/'spider_amr_features.csv', index=False)
    #
    # # paws_amr = pd.read_csv(data_dir/'paws_amr.csv')
    # paws = pd.read_csv(tct_out_dir/'paws_text_features.csv')
    # paws = get_amr_features_two_sent(paws, amr_col1 = 'premise_amr', amr_col2 = 'hypothesis_amr')
    # paws = get_3_amr_features(paws, amr_pred='premise_amr', amr_gold='hypothesis_amr')
    # paws.to_csv(tct_out_dir/'paws_features.csv', index=False)

    amrs = pd.read_csv(f'{google_pred_dir}/corrected_amrs.csv')

    # asilm = pd.read_csv(tct_out_dir/'asilm_text_features.csv')
    # asilm['id'] = asilm['id_y']
    # for index, row in asilm.iterrows():
    #     asilm.loc[index, 'premise_amr'] = amrs[amrs['id'] == f"{row['id']}_p"]['amr'].values[0]
    #     asilm.loc[index, 'hypothesis_amr'] = amrs[amrs['id'] == f"{row['id']}_h"]['amr'].values[0]
    #
    #
    # asilm = get_amr_features_two_sent(asilm, amr_col1 = 'premise_amr', amr_col2 = 'hypothesis_amr')
    # asilm = get_3_amr_features(asilm, amr_pred='premise_amr', amr_gold='hypothesis_amr')
    # asilm.to_csv(tct_out_dir/'asilm_features.csv', index=False)

    # wmt = pd.read_csv(tct_out_dir/'wmt_text_features.csv')
    # for index, row in wmt.iterrows():
    #     wmt.loc[index, 'en_amr'] = amrs[amrs['id'] == f"{row['id']}_en"]['amr'].values[0]
    # wmt = get_amr_features_one_sent(wmt, amr_col = 'en_amr')
    # wmt.to_csv(tct_out_dir/'wmt_amr_features.csv', index=False)

    # logic = pd.read_csv(tct_out_dir/'logic_text_features.csv')
    # logic['id'] = logic['id_y']
    # for index, row in logic.iterrows():
    #     logic.loc[index, 'amr'] = amrs[amrs['id'] == f"{row['id']}"]['amr'].values[0]
    #
    # logic = get_amr_features_one_sent(logic, amr_col = 'amr')
    # logic.to_csv(tct_out_dir/'logic_amr_features.csv', index=False)


    # pubmed = pd.read_csv(tct_out_dir/'pubmed45_text_features.csv')
    # if 'id_y' in pubmed.columns:
    #     pubmed['id'] = pubmed['id_y']
    # for index, row in pubmed.iterrows():
    #     pubmed.loc[index, 'amr'] = amrs[amrs['id'] == f"{row['id']}"]['amr'].values[0]
    #
    # pubmed = get_amr_features_one_sent(pubmed, amr_col = 'amr')
    # pubmed.to_csv(tct_out_dir/'pubmed45_features.csv', index=False)
    #
    #
    # spider = pd.read_csv(tct_out_dir/'spider_text_features.csv')
    # if 'id_y' in spider.columns:
    #     spider['id'] = spider['id_y']
    # for index, row in spider.iterrows():
    #     spider.loc[index, 'amr'] = amrs[amrs['id'] == f"{row['id']}"]['amr'].values[0]
    #
    # spider = get_amr_features_one_sent(spider, amr_col = 'amr')
    # spider.to_csv(tct_out_dir/'spider_amr_features.csv', index=False)

    # ldc_ner = pd.read_csv(tct_out_dir/'ldc_ner_text_features.csv')
    # if 'id_y' in ldc_ner.columns:
    #     ldc_ner['id'] = ldc_ner['id_y']
    # for index, row in ldc_ner.iterrows():
    #     ldc_ner.loc[index, 'amr'] = amrs[amrs['id'] == f"{row['id']}"]['amr'].values[0]
    #
    # ldc_ner = get_amr_features_one_sent(ldc_ner, amr_col = 'amr')
    # ldc_ner.to_csv(tct_out_dir/'ldc_ner_amr_features.csv', index=False)
    scoring_gpt_smatch()


if __name__ == '__main__':
    # replace invalid amr with retried amr
    # full= pd.read_csv(output_dir/'gpt-3.5-turbo-0613_ldc_test_amrs_full.csv')
    # retry = pd.read_csv(data_dir/'gpt-3.5-turbo-0613_ldc_test_amrs.csv')
    # for i, d in full.iterrows():
    #     if d['invalid_amr'] == 1:
    #         full.iloc[i] = retry[retry['id'] == d['id']].values[0]
    main()
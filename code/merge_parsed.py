import pandas as pd
from pathlib import Path
import os
import random
import re

data_dir = Path(__file__).parent.parent.resolve() / "data"
parsed_dir  = data_dir / "parsed_amrs"

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

combined = []
df_all = pd.DataFrame()
for file in os.listdir(parsed_dir):
    if file.endswith(".csv") and 'all' not in file:
        print(file)
        df = pd.read_csv(parsed_dir / file, header=None)
        df.columns = ['id','text','AMR3_structbart_L_amr']
        if 'bio' in file:
            if 'train' in file:
                df['split'] = 'train'
            elif 'dev' in file:
                df['split'] = 'dev'
            elif 'test' in file:
                df['split'] = 'test'
            df['source_set'] ='bio_amr0.8'
        elif 'ldc' in file:
            df['split'] = df['id'].apply(lambda x: x.split("_")[-2])
            df['source_set'] = 'ldc_amr3.0'

        print(df.head())
        df_all = pd.concat([df_all, df], axis = 0)
df_all['text_detok'] = df_all['text']
df_all['AMR3_structbart_L_amr'] = df_all['AMR3_structbart_L_amr'].apply(lambda x: "(" + x.split("\n(")[-1])
df_all.to_csv(data_dir / "parsed_amrs" / "all_amrs.csv", index=False)
print(df_all.shape)


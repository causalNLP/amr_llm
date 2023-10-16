
import pandas as pd
import datasets
from sklearn.svm import SVC
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,\
Trainer, TrainingArguments,AutoTokenizer, AutoModel, AutoModelForSequenceClassification,\
AdamW, AutoTokenizer,get_linear_schedule_with_warmup, AutoConfig, PreTrainedModel,\
get_linear_schedule_with_warmup
import transformers
from sentence_transformers import SentenceTransformer, InputExample, losses
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut, train_test_split, KFold
from tqdm import tqdm
import os
# from oauth2client.client import GoogleCredentials
# import gspread
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from efficiency.function import random_sample, set_seed
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import ast
from pathlib import Path
import random
from tqdm import tqdm


def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except ValueError:
        return val  # return the original value if it can't be parsed

def embed(df, input_col1, input_col2):
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Generate embeddings for premise and hypothesis
    premise_embeddings = model.encode(df[input_col1].tolist())
    hypothesis_embeddings = model.encode(df[input_col2].tolist())
    # Combine the premise and hypothesis embeddings
    # You could also try other methods of combination such as subtraction or multiplication
    embeddings = [np.concatenate((p, h)) for p, h in zip(premise_embeddings, hypothesis_embeddings)]
    premise_embeddings_list = [safe_literal_eval(embedding.tolist()) for embedding in premise_embeddings]
    hypothesis_embeddings_list = [safe_literal_eval(embedding.tolist()) for embedding in hypothesis_embeddings]
    # Add the embeddings as new columns in the dataframe
    cos_sim_embed = [cosine_similarity(p, h) for p, h in zip(premise_embeddings_list, hypothesis_embeddings_list)]
    df['cos_sim_embed'] = cos_sim_embed
    return df


def main():
    np.random.seed(seed=0)
    torch.manual_seed(0)
    np.random.seed(0)
    set_seed(0)
    random.seed(0)

    root_dir = Path(__file__).parent.parent.resolve()
    data_dir = root_dir / "data"
    feature_dir = data_dir / "featured"
    model_dir = root_dir / "model"
    to_process_dir = data_dir / "to_process"
    tct_out_dir = data_dir / "tct_outputs"

    # model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save(model_dir / "sent_trans")
    tokenizer.save_pretrained(model_dir / "sent_trans_tokenizer")

if __name__ == "__main__":
    main()

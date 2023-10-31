import argparse
import logging
import os
import sys
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from models import DfModel as DM
from models import data_preparation
from utils import log_args, add_args, LoggerWritter, modify_dict_from_dataparallel
# from analysis import ErrorAnalysis as ALS
from available_models import all_models


def eval(args):

    dm = data_preparation(args, log, mode='eval')

    log.info('Get tokenizer:')
    dm.set_device()
    dm.get_tokenizer()

    dm.dataloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False
    }

    for filename in args.splits_filename:
        log.info(f'\nStart evaluation on file: {filename}')

        log.info('Get dataloader:')
        dm.df_test = pd.read_csv(filename, engine='python')
        dm.get_dataloader(mode='eval')

        dm.args.dropout_rate_curr = args.dropout_rate
        if dm.args.output_type in ['binary', 'categorical']:
            dm.loss_f = torch.nn.CrossEntropyLoss()
        elif dm.args.output_type == 'real':
            dm.loss_f = torch.nn.MSELoss()

        log.info('Get model..')
        dm.get_model()

        log.info('Get model checkpoint..')
        # if loading a state_dict saved without removing extra keys in the dict,
        dm.model.load_state_dict(modify_dict_from_dataparallel(
            torch.load(args.model_load_path), args))
        # dm.model.load_state_dict(torch.load(args.model_load_path))
        dm.model.eval()

        log.info('Begin evaluation:')
        dm.eval(split='test', store_csv=True, report_analysis=True)

        log.info(f'End evaluation on file: {filename}\n')

        # log.info('Begin error analysis:')
        # als = ALS(args)
        # als.get_classification_report()
        # als.get_classification_heatmap()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # evaluation
    # splits_filename: a list of files that needs to be evaluated on.
    parser.add_argument('--splits_filename', nargs='*',
                        type=str, default=['data/example_data.csv'])

    parser.add_argument('--text_col', type=str,
                        default='sentence_deleted_hedge')
    parser.add_argument('--y_col', type=str, default='uncertainty_output')
    parser.add_argument('--num_numeric_features', type=int, default=10)
    parser.add_argument('--numeric_features_col',
                        nargs='*', type=str, default=[
                            'word_number', 'sentence_placement', 'sentence_number', 'avg_sentence_length', 'averagecitations', 'numberauthors', 'yearPub', 'weighteda1hindex', 'weightedi10index', 'female_ratio'
                        ])

    # macro settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='flexible',
                        choices=['flexible', 'cpu'])
    parser.add_argument('--model_load_path', type=str, default='models/trained_scibert_uncertainty.pt')
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--csv_output_path', type=str,
                        default='output/test_example.csv')  # .csv
    parser.add_argument('--dataparallel', type=bool, default=True)

    # training task related
    parser.add_argument('--output_type', type=str, default='real',
                        choices=['binary', 'categorical', 'real'])
    # if output_type is categorical
    parser.add_argument('--num_classes', type=int, default=None)

    # pretrained model related
    parser.add_argument('--max_length', type=int, default=512,
                        help='the input length for bert')
    parser.add_argument('--pretrained_model', type=str, default='allenai/scibert_scivocab_uncased',
                        choices=list(all_models.keys()))

    # hyperparams below will have accompanying current value as attributes in args.
    # For example, <batch_size> will have accompanying value <batch_size_curr> in the actual training.
    # list for searching hyperms
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    # list for searching hyperms, a list of list
    parser.add_argument('--hidden_dim_curr', nargs='*', type=int, default=[50])

    # error analysis
    parser.add_argument('--img_output_dir', type=str, default=None)
    # regression
    # parser.add_argument('--')

    args = parser.parse_args()
    args.dataset_class_dir = None

    # logging
    now = datetime.now()
    now = now.strftime("%Y%m%d-%H:%M:%S")
    args.now = now
    handler = logging.FileHandler(filename=f'{args.log_dir}eval-{now}.log')
    log = logging.getLogger('bert_tune')
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    sys.stderr = LoggerWritter(log.warning)

    log_args(args, log)
    args = add_args(args)
    eval(args)
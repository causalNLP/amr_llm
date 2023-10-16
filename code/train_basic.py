import os
import json
from sklearn.impute import SimpleImputer
import math
import random
import re
from tqdm import tqdm
import pandas as pd
import itertools
from efficiency.function import random_sample, set_seed
import string
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, GridSearchCV,KFold
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.linear_model import LogisticRegression, LinearRegression,Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, make_scorer, confusion_matrix
import matplotlib as plt
from collections import Counter
from numpy import dot
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error

import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from tkinter.constants import E
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import RFECV
from numpy import loadtxt
from xgboost import XGBClassifier
from pathlib import Path
np.random.seed(0)
set_seed(0)
random.seed(0)

root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
tct_out_dir = data_dir / "tct_outputs"
parent_dir = os.path.dirname(root_dir)
good_dir = data_dir / "good"
prediction_dir = data_dir / "predictions"
onto_dir = f'{parent_dir}/ontonotes-release-5.0'
model_dir = root_dir / "model"
sample_dir = data_dir / "samples"



class Train():
  def __init__(self, df, target = ['amr_improve'], model_type = "LogisticRegression", svc_args = {'kernel': 'linear'}):
    self.df = df
    self.target = target
    self.model_type = model_type
    if self.model_type in ["LinearRegression", "Lasso", "Ridge"]:
        self.y = df[self.target]
    else:
        self.y = self.df[self.target].apply(lambda x: int(x>0))
        self.df[self.target] = self.y

    self.svc_args = svc_args

  def get_features(self, df = None, features=[], to_exclude=['truth']):
    if df is None:
        df = self.df
    to_drop = ['premise', 'hypothesis', 'pred', 'query', 'pred_norm', 'score',
               'query0', 'pred0', 'query1', 'pred1', 'query2', 'pred2', 'query3',
               'pred3', 'pred_norm_amr', 'score_amr', 'amr_impact', 'bow_pre', 'bow_hyp', 'amr_improve', 'gpt_fail',
               'did_llm_fail', 'did_llm_failed','pure_helpfulness',
               'ground_truth', 'llm_ouput_postprocessed', 'amrCoT_ouput_postprocessed',
               'did_amr_help', 'helpfulness', 'llm_direct', 'bleu', 'pred_amr', 'bleu_amr', 'amrCoT_output',
               'amrCoT_output_postprocessed','f1','f1_amr','f1_amrcot','sentence' 'interaction' 'interaction_len' 'interaction_tfidf'
 'interaction_distance' 'sentence_amr' 'text' 'text_amr' 'invalid_type']
    if not len(features):
      to_drop += to_exclude
      # Drop non-numerical columns
      X = df.select_dtypes(include=[np.number])
      # Drop columns with only one value
      X = df.loc[:, df.apply(pd.Series.nunique) != 1]
      for column_name in df.columns:
        if 'id' == column_name:
          to_drop.append(column_name)
        elif 'Unnamed' in column_name:
          to_drop.append(column_name)
        elif 'label' in column_name:
            to_drop.append(column_name)

        elif not pd.api.types.is_numeric_dtype(df[column_name]):
          to_drop.append(column_name)

      to_drop = [col for col in to_drop if col in df.columns]
      X = df.drop(columns=to_drop)
      return X

    X = df[features]
    return X

  def split_sets(self, df,  dataset = 'all'):
      """Split data into train, dev and test sets, formatting depends on the dataset"""

      if dataset in ['all']:
          # Split the data into 70% train and 30% temporary set (will be split into dev and test)
          train_set, temp_set = train_test_split(df, test_size=0.3, random_state=42)

          # Split the temporary set into 50% dev and 50% test (making them 15% of the total each)
          dev_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)

      elif dataset in ['translation']:
          df['set'] = df.id.str[:10]
          train_set = df.loc[df['set'] == 'newstest13']
          dev_set, test_set = train_test_split(df.loc[df['set'] == 'newstest16'], test_size=0.5, random_state=42)
      elif dataset in ['PAWS', 'pubmed']:
          train_set, val_df = train_test_split(df, test_size=0.3, random_state=42)
          dev_set, test_set = train_test_split(val_df, test_size=0.5, random_state=42)
      elif dataset in ['logic', 'django', 'spider']:
          train_set = df.loc[df['id'].str.contains('train')]
          test_set = df.loc[df['id'].str.contains('test')]
          dev_set = df.loc[df['id'].str.contains('dev')]


      return train_set, dev_set, test_set

  def tt_split(self, split_by = None, test_criterion = None, dev_criterion = None,  val_size=0.1, test_size=0.1):
    dataset = self.df['id'].iloc[0].split('_')[0]
    if len(self.df) > 10000:
        dataset = 'all'
    if 'newstest' in dataset:
        dataset = 'translation'
    elif 'pubmed45' in dataset:
        dataset = 'pubmed'
    elif 'paws' in dataset:
        dataset = 'PAWS'
    # First, split the data into train+val and test sets
    if dataset in ['translation','PAWS','pubmed','logic','django','spider','all']:
        df_train, df_val, df_test = self.split_sets(self.df, dataset)
    else:
        if split_by:
          df_train_val = self.df[~self.df[split_by].str.contains(test_criterion)]
          df_test = self.df[self.df[split_by].str.contains(test_criterion)]
        if dev_criterion:
          df_train = df_train_val[~df_train_val[split_by].str.contains(dev_criterion)]
          df_val = df_train_val[df_train_val[split_by].str.contains(dev_criterion)]
        else:
          df_train, df_val_test = train_test_split(self.df, test_size=test_size+val_size, random_state=0)
          df_val, df_test = train_test_split(df_val_test, test_size=test_size/(val_size+test_size), random_state=0)
    print("Train size: ", len(df_train))
    print("Val size: ", len(df_val))
    print("Test size: ", len(df_test))
    X_train = self.get_features(df_train)
    X_val = self.get_features(df_val)
    X_test = self.get_features(df_test)
    y_train = df_train[self.target]
    y_val = df_val[self.target]
    y_test = df_test[self.target]
    return X_train, y_train, X_val, y_val, X_test, y_test

  def scale(self, X):
      # Check for NaN values
      if X.isna().any().any():
          # If there are NaN values, replace them with the median
          imputer = SimpleImputer(strategy='median')
          X = imputer.fit_transform(X)

      # Scale the data
      scaler = RobustScaler()
      X_scaled = scaler.fit_transform(X)

      return X_scaled


  def train(self,split_by = None, test_criterion = None, dev_criterion = None, max_f1 = True):
    print("Training model: ", self.model_type)
    if self.model_type == "Random":
        X_train, y_train, X_val, y_val, X_test, y_test = self.tt_split(split_by=split_by, test_criterion=test_criterion, dev_criterion=dev_criterion, val_size=0.1, test_size=0.1)
        all_classes = np.unique(y_train)
        # Randome classifier that always predicts the majority class
        major_class = np.argmax(np.bincount(y_train))
        y_pred = np.full(len(y_test), major_class)
        report = classification_report(y_test, y_pred,  digits=4, zero_division=0)
        print("Random classifier: Majority class")
        print(report)

        # Random classifier that predicts randomly with the same distribution as the training set
        y_pred_random_weighted = np.random.choice(all_classes, size = len(y_test), p = np.bincount(y_train) / len(y_train))
        report = classification_report(y_test, y_pred_random_weighted, digits=4, zero_division=0)
        print("Random classifier: Weighted random")
        print(report)


        # Random classifier that predicts randomly
        all_classes = np.unique(y_train)
        y_pred_random = np.random.choice(all_classes, size = len(y_test))
        report = classification_report(y_test, y_pred_random, digits=4, zero_division=0)
        print("Random classifier: Uniformly Random")
        print(report)
        return




        # Apply 5-fold cross-validation
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(self.df):
            X_train, X_test = self.df.iloc[train_index], self.df.iloc[test_index]
            y_train, y_test = self.df.iloc[train_index], self.df.iloc[test_index]
            # Scale the data
            X_train_scaled = torch.tensor(self.scale(X_train), dtype=torch.float32)
            X_test_scaled = torch.tensor(self.scale(X_test), dtype=torch.float32)
            # Continue with your training code...
    else:
        # Apply train-test split
        X_train, y_train, X_val, y_val, X_test, y_test = self.tt_split(split_by=split_by, test_criterion=test_criterion, dev_criterion=dev_criterion, val_size=0.1, test_size=0.1)
        X_train_scaled = torch.tensor(self.scale(X_train), dtype=torch.float32)
        X_val_scaled = torch.tensor(self.scale(X_val), dtype=torch.float32)
        X_test_scaled = torch.tensor(self.scale(X_test), dtype=torch.float32)
        # Continue with your training code...


    if len(self.df) < 600:
        # perform 5-fold cross validation on X_train_scaled and y_train
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        print('Performing 5-fold cross validation on training set')
        f1_scores = []
        best_thresholds = []
        models = []
        for train_index, test_index in kf.split(X_train_scaled):
            X_train_temp, X_test_temp = X_train_scaled[train_index], X_train_scaled[test_index]
            y_train_temp, y_test_temp = y_train.iloc[train_index], y_train.iloc[test_index]

            if self.model_type == "LogisticRegression":
                clf = LogisticRegression(max_iter=5000, class_weight="balanced", penalty=None, random_state=0).fit(
                    X_train_temp, y_train_temp)
            elif self.model_type == "LinearRegression":
                clf = LinearRegression(fit_intercept=True)
                # Use both train and dev to train the regression model
                X_train_scaled = torch.cat((X_train_scaled, X_val_scaled), 0)
                y_train = pd.concat([y_train, y_val], axis=0)
                clf.fit(X_train_scaled, y_train)
                clf.fit(X_train_temp, y_train_temp)
            elif self.model_type == "Lasso":
                clf = LinearRegression(fit_intercept=True)
                clf.fit(X_train_temp, y_train_temp)
            elif self.model_type == "Ridge":
                clf = LinearRegression(fit_intercept=True)
                clf.fit(X_train_temp, y_train_temp)
            elif self.model_type == "DecisionTree":
                clf = DecisionTreeClassifier(class_weight="balanced", random_state=0).fit(X_train_temp, y_train_temp)
            elif self.model_type == "RandomForest":
                clf = RandomForestClassifier(class_weight="balanced", random_state=0).fit(X_train_temp, y_train_temp)
            elif self.model_type == "XGBoost":
                count_class_0 = sum(y_train_temp == 0)
                count_class_1 = sum(y_train_temp == 1)
                scale_pos_weight = count_class_0 / count_class_1
                clf = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=0).fit(X_train_temp, y_train_temp)
            elif self.model_type == "SVM":
                # Create an SVM classifier with a linear kernel
                clf = SVC(**self.svc_args, probability=True, class_weight="balanced", random_state=0).fit(
                    X_train_temp, y_train_temp)
            elif self.model_type == "Ensemble":
                count_class_0 = sum(y == 0)
                count_class_1 = sum(y == 1)
                scale_pos_weight = count_class_0 / count_class_1
                estimators = [
                    ('rf', RandomForestClassifier(class_weight="balanced", random_state=0)),
                    ('xgb', XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=0)),
                    ('svm_poly', SVC(kernel='poly', probability=True, class_weight="balanced", random_state=0)),
                    ('gdb', GradientBoostingClassifier(random_state=0)),
                    # ('svm_rbf', SVC(kernel = 'rbf', gamma = 'auto', probability = True, class_weight = "balanced", random_state=0)),
                ]
                clf = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression()
                )
                clf.fit(X_train_temp, y_train_temp)


            y_test_temp_probs = clf.predict_proba(X_test_temp)[:, 1]
            thresholds = np.linspace(0, 1, 100)
            f1_scores_current = [f1_score(y_test_temp, y_test_temp_probs > t, pos_label=True, zero_division=0) for t in thresholds]
            best_threshold = thresholds[np.argmax(f1_scores_current)]

            # Calculate the F1 score of the model with the best threshold on the validation set
            y_val_probs = clf.predict_proba(X_val_scaled)[:, 1]
            y_val_pred = y_val_probs > best_threshold
            f1_score_val = f1_score(y_val, y_val_pred, pos_label=True, zero_division=0)
            print('F1 score on validation set: ', f1_score_val)
            # save the model, best threshold and f1 score on validation set
            models.append(clf)
            f1_scores.append(f1_score_val)
            best_thresholds.append(best_threshold)

        # After 5 folds CV, choose the model with the highest F1 score on the validation set
        best_model_index = np.argmax(f1_scores)
        best_threshold = best_thresholds[best_model_index]

        print('Best model index: ', best_model_index)
        print('Best threshold: ', best_threshold)

        # Choose the best model
        clf = models[best_model_index]
        # Calculate the classification report on the test set
        y_test_probs = clf.predict_proba(X_test_scaled)[:, 1]
        y_test_pred = y_test_probs > best_threshold
        report = classification_report(y_test, y_test_pred, digits=4, zero_division=0)
        print("Best_threshold: ", best_threshold)
        print(report)
        # print("Random classifier:\n ",classification_report(y_test, y_pred_random, digits=4))
        return clf, report

    else:
        if self.model_type == "LogisticRegression":
            clf = LogisticRegression(max_iter=5000, class_weight="balanced", penalty = None, random_state=0).fit(X_train_scaled, y_train)
        elif self.model_type == "LinearRegression":
            clf = LinearRegression(fit_intercept=True)
            # Use both train and dev to train the regression model
            X_train_scaled = torch.cat((X_train_scaled, X_val_scaled), 0)
            y_train = pd.concat([y_train, y_val], axis=0)
            clf.fit(X_train_scaled, y_train)
        elif self.model_type == "Lasso":
            clf = Lasso(fit_intercept=True)
            clf.fit(X_train_scaled, y_train)
            X_train_scaled = torch.cat((X_train_scaled, X_val_scaled), 0)
            y_train = pd.concat([y_train, y_val], axis=0)
            clf.fit(X_train_scaled, y_train)
        elif self.model_type == "Ridge":
            clf = Ridge(fit_intercept=True)
            clf.fit(X_train_scaled, y_train)
            X_train_scaled = torch.cat((X_train_scaled, X_val_scaled), 0)
            y_train = pd.concat([y_train, y_val], axis=0)
            clf.fit(X_train_scaled, y_train)
        elif self.model_type == "DecisionTree":
            clf = DecisionTreeClassifier(class_weight="balanced", random_state=0).fit(X_train_scaled, y_train)
        elif self.model_type == "RandomForest":
            clf = RandomForestClassifier(class_weight="balanced", random_state=0).fit(X_train_scaled, y_train)
        elif self.model_type == "XGBoost":
          count_class_0 = sum(y_train == 0)
          count_class_1 = sum(y_train == 1)
          scale_pos_weight = count_class_0 / count_class_1
          clf = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=0).fit(X_train_scaled, y_train)
        elif self.model_type == "SVM":
          # Create an SVM classifier with a linear kernel
          clf = SVC(**self.svc_args, probability = True, class_weight = "balanced", random_state=0).fit(X_train_scaled, y_train)
        elif self.model_type == "Ensemble":
          count_class_0 = sum(y_train == 0)
          count_class_1 = sum(y_train == 1)
          scale_pos_weight = count_class_0 / count_class_1
          estimators = [
              ('rf', RandomForestClassifier(class_weight="balanced", random_state=0)),
              ('xgb', XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=0)),
              # ('svm_poly', SVC(kernel = 'poly', probability = True, class_weight = "balanced", random_state=0)),
              ('gdb', GradientBoostingClassifier(random_state=0)),
              # ('svm_rbf', SVC(kernel = 'rbf', gamma = 'auto', probability = True, class_weight = "balanced", random_state=0)),
          ]
          clf = StackingClassifier(
              estimators=estimators,
              final_estimator=LogisticRegression()
          )
          clf.fit(X_train_scaled, y_train)


    # Find the best threshold using the validation set
    if self.model_type in ["LinearRegression",'Lasso','Ridge']:
        print(f'{self.model_type} model:')
        # print(the model coefficients, training and test scores)
        print("Intercept: ", clf.intercept_)
        # print('Coefficients: \n', clf.coef_)
        print(X_train_scaled.shape)
        print(len(clf.coef_))
        print('Training score: ', clf.score(X_train_scaled, y_train))
        print('Test score: ', clf.score(X_test_scaled, y_test))
        y_pred = clf.predict(X_test_scaled)
        loss = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {loss}')
        return clf, None
    if max_f1:
        y_val_probs = clf.predict_proba(X_val_scaled)[:, 1]
        thresholds = np.linspace(0, 1, 100)
        f1_scores = [f1_score(y_val, y_val_probs > t, pos_label=True, zero_division=0) for t in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
    else:
        best_threshold = 0.5

    # Evaluate the model on the test set using the best threshold
    y_test_probs = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_test_probs > best_threshold).astype(int)



    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    print("Best_threshold: ", best_threshold)
    print(report)
    # print("Random classifier:\n ",classification_report(y_test, y_pred_random, digits=4))
    return clf, report


def train_para(model_type="XGBoost"):
  print('Start training para')
  ### Paraphrases Detection
  pred_file1 = f"{google_pred_dir}/final_results_paws.csv"
  # pred_file2 = f"{google_pred_dir}/final_results_paws_dev.csv"
  pred = pd.read_csv(pred_file1)
  # pred2 = pd.read_csv(pred_file2)
  # pred = pd.concat([pred1, pred2])
  # pred['id'] = pred['id__'].apply(lambda s: s[:-1])
  feature_file = good_dir / "paws_features.csv"
  df = pd.read_csv(feature_file)
  different = False
  if 'id_y' in df.columns:
    df['id'] = df['id_y']
    df = df.drop(['id_x', 'id_y'], axis=1)
    different = True

  for col in df.columns:
      if '.1' in col or '.2' in col or '.3' in col or 'Unnamed' in col or 'helpfulness' in col:
          df.drop(col, axis=1, inplace=True)
          different = True
  if different:
      df.to_csv(feature_file, index=False)
  if 'helpfulness' in df.columns:
      df = df.drop('helpfulness', axis=1)
  print(pred.columns)
  df_full = df.merge(pred[['id', 'helpfulness']], on="id")
  df_full['did_amr_help'] = df_full['helpfulness']
  df = df_full.copy()
  df['amr_improve'] = df['did_amr_help'].apply(lambda x: x > 0)

  trainer = Train(df, target='amr_improve', model_type= model_type)
  if model_type == 'Random':
    trainer.train(split_by='id', test_criterion='paws_test', max_f1=True)
    return
  xgb, report = trainer.train(split_by='id', test_criterion='paws_test', max_f1=True)


def train_translation(model_type = "XGBoost"):
  ### Translation
  print('Start training translation')
  pred_file = f"{google_pred_dir}/final_results_trans_corrected.csv"
  pred = pd.read_csv(pred_file)
  pred['id'] = pred['id'].apply(lambda s: s[:-1])
  feature_file = good_dir / "wmt_features.csv"
  df = pd.read_csv(feature_file)
  different = False
  if 'id_y' in df.columns:
    df['id'] = df['id_y']
    df = df.drop(['id_x', 'id_y'], axis=1)
    different = True

  for col in df.columns:
      if '.1' in col or '.2' in col or '.3' in col or 'Unnamed' in col or 'helpfulness' in col:
          df.drop(col, axis=1, inplace=True)
          different = True

  if different:
      df.to_csv(feature_file, index=False)


  df_full = df.merge(pred[['id','helpfulness']], on = "id", how ='inner')
  df_full['did_amr_help'] = df_full['helpfulness']
  df = df_full.copy()
  df['amr_improve'] = df['did_amr_help'].apply(lambda x: x > 0)
  # X = get_features(df)
  # y = df['amr_improve']
  trainer = Train(df,target ='amr_improve', model_type = model_type)
  if model_type == 'Random':
    trainer.train(split_by='id', test_criterion='newstest16', max_f1=True)
    return
  xgb, report = trainer.train(split_by='id', test_criterion = 'newstest16', max_f1 = True)



def train_LOGIC(model_type = "XGBoost"):
  print('Start training LOGIC')
  ### Logic Fallacy Detection
  pred_file = f"{google_pred_dir}/final_results_logic_corrected.csv"
  feature_file = good_dir / "logic_features.csv"

  pred = pd.read_csv(pred_file)
  df = pd.read_csv(feature_file)
  different = False
  if 'id_y' in df.columns:
    df['id'] = df['id_y']
    df = df.drop(['id_x', 'id_y'], axis=1)
    different = True

  for col in df.columns:
      if '.1' in col or '.2' in col or '.3' in col or 'Unnamed' in col or 'helpfulness' in col:
          df.drop(col, axis=1, inplace=True)
          different = True
  if different:
      df.to_csv(feature_file, index=False)


  df_full = df.merge(pred[['id','helpfulness']], on = "id")
  df_full['did_amr_help'] = df_full['helpfulness']
  df = df_full.copy()
  df['amr_improve'] = df['did_amr_help'].apply(lambda x: x > 0)
  trainer = Train(df,target ='amr_improve', model_type = model_type)
  if model_type == 'Random':
    trainer.train(split_by='id', test_criterion='logic_test', dev_criterion='logic_dev', max_f1=True)
    return
  else:
    xgb, report = trainer.train(split_by='id', test_criterion = 'logic_test', dev_criterion = 'logic_dev', max_f1 = True)

def train_pubmed(model_type = "XGBoost"):
    print('Start training pubmed')
    ### Pubmed
    pred_file = f"{google_pred_dir}/final_results_pubmed_corrected.csv"
    feature_file = good_dir/"pubmed45_features.csv"
    pred = pd.read_csv(pred_file)
    df = pd.read_csv(feature_file)
    different = False
    if 'id_y' in df.columns:
        df['id'] = df['id_y']
        df = df.drop(['id_x', 'id_y'], axis=1)
        different = True

    for col in df.columns:
        if '.1' in col or '.2' in col or '.3' in col or 'Unnamed' in col or 'helpfulness' in col:
            df.drop(col, axis=1, inplace=True)
            different = True
    if different:
        df.to_csv(feature_file, index=False)

    df_full = df.merge(pred[['id','helpfulness']], on = "id")
    df_full['did_amr_help'] = df_full['helpfulness']
    df = df_full.copy()
    df['amr_improve'] = df['did_amr_help'].apply(lambda x: x > 0)
    trainer = Train(df,target ='amr_improve', model_type = model_type)
    if model_type == 'Random':
        trainer.train(max_f1=True)
        return
    else:
        xgb, report = trainer.train(max_f1 = True)

def train_asilm(model_type = "XGBoost"):
    pred_file = f"{google_pred_dir}/final_results_asilm_corrected.csv"
    feature_file = good_dir / "asilm_features.csv"
    pred = pd.read_csv(pred_file)
    df = pd.read_csv(feature_file)
    different = False
    if 'id_y' in df.columns:
        df['id'] = df['id_y']
        df = df.drop(['id_x', 'id_y'], axis=1)
        different = True

    for col in df.columns:
        if '.1' in col or '.2' in col or '.3' in col or 'Unnamed' in col or 'helpfulness' in col:
            df.drop(col, axis=1, inplace=True)
            different = True
    if different:
        df.to_csv(feature_file, index=False)

    df_full = df
    df_full['did_amr_help'] = df_full['helpfulness']
    df = df_full.copy()
    df['amr_improve'] = df['did_amr_help'].apply(lambda x: x > 0)
    trainer = Train(df,target ='amr_improve', model_type = model_type)
    if model_type == 'Random':
        trainer.train(max_f1=True,)
        return
    else:
        xgb, report = trainer.train(max_f1 = True)


def train_ldc_slang_parser(model_type = "XGBoost"):
    print('Start training ldc_slang_parser')
    ### Gold Slang
    pred_file = f"{google_pred_dir}/final_results_paraphrase_slang_gold_parser.csv"
    feature_file = good_dir / "lde_slang_features_parser.csv"
    pred = pd.read_csv(pred_file)
    df = pd.read_csv(feature_file)
    different = False
    if 'id_y' in df.columns:
        df['id'] = df['id_y']
        df = df.drop(['id_x', 'id_y'], axis=1)
        different = True

    for col in df.columns:
        if '.1' in col or '.2' in col or '.3' in col or 'Unnamed' in col or 'helpfulness' in col:
            df.drop(col, axis=1, inplace=True)
            different = True
    if different:
        df.to_csv(feature_file, index=False)

    df_full = df.merge(pred[['id','helpfulness']], on = "id")
    df_full['did_amr_help'] = df_full['helpfulness']
    df = df_full.copy()
    df['amr_improve'] = df['did_amr_help'].apply(lambda x: x > 0)
    trainer = Train(df,target ='amr_improve', model_type = model_type)
    if model_type == 'Random':
        trainer.train(max_f1=True)
        return
    else:
        xgb, report = trainer.train(max_f1 = True)

def train_ldc_slang_gold(model_type = "XGBoost"):
    print('Start training ldc_slang_true')
    ### Gold Slang
    pred_file = f"{google_pred_dir}/final_results_ldc_slang_gold.csv"
    feature_file = good_dir / "lde_slang_features_true.csv"
    pred = pd.read_csv(pred_file)
    df = pd.read_csv(feature_file)
    different = False
    if 'id_y' in df.columns:
        df['id'] = df['id_y']
        df = df.drop(['id_x', 'id_y'], axis=1)
        different = True

    for col in df.columns:
        if '.1' in col or '.2' in col or '.3' in col or 'Unnamed' in col or 'helpfulness' in col:
            df.drop(col, axis=1, inplace=True)
            different = True
    if different:
        df.to_csv(feature_file, index=False)

    df_full = df.merge(pred[['id','helpfulness']], on = "id")
    df_full['did_amr_help'] = df_full['helpfulness']
    df = df_full.copy()
    df['amr_improve'] = df['did_amr_help'].apply(lambda x: x > 0)
    trainer = Train(df,target ='amr_improve', model_type = model_type)
    if model_type == 'Random':
        trainer.train(max_f1=True)
        return
    else:
        xgb, report = trainer.train(max_f1 = True)

def train_django(model_type = "XGBoost"):
    print('Start training django')
    ### Django
    pred_file = f"{google_pred_dir}/final_results_django_corrected.csv"
    feature_file = good_dir/ "django_features.csv"
    pred = pd.read_csv(pred_file)
    df = pd.read_csv(feature_file)
    different = False
    if 'id_y' in df.columns:
        df['id'] = df['id_y']
        df = df.drop(['id_x', 'id_y'], axis=1)
        different = True

    for col in df.columns:
        if '.1' in col or '.2' in col or '.3' in col or 'Unnamed' in col or 'helpfulness' in col:
            df.drop(col, axis=1, inplace=True)
            different = True
    if different:
        df.to_csv(feature_file, index=False)
    df_full = df.merge(pred[['id','helpfulness']], on = "id")
    df_full['did_amr_help'] = df_full['helpfulness']
    df = df_full.copy()
    df['amr_improve'] = df['did_amr_help'].apply(lambda x: x > 0)
    trainer = Train(df,target ='amr_improve', model_type = model_type)
    if model_type == 'Random':
        trainer.train(max_f1=True)
        return
    else:
        xgb, report = trainer.train(max_f1 = True)

def train_spider(model_type = "XGBoost"):
    print('Start training spider')
    ### Spider
    pred_file = f"{google_pred_dir}/final_results_spider_corrected.csv"
    feature_file = good_dir / "spider_features.csv"
    pred = pd.read_csv(pred_file)
    df = pd.read_csv(feature_file)
    different = False
    if 'id_y' in df.columns:
        df['id'] = df['id_y']
        df = df.drop(['id_x', 'id_y'], axis=1)
        different = True

    for col in df.columns:
        if '.1' in col or '.2' in col or '.3' in col or 'Unnamed' in col or 'helpfulness' in col:
            df.drop(col, axis=1, inplace=True)
            different = True
    if different:
        df.to_csv(feature_file, index=False)

    df_full = df.merge(pred[['id','helpfulness']], on = "id")
    df_full['did_amr_help'] = df_full['helpfulness']
    df = df_full.copy()
    df['amr_improve'] = df['did_amr_help'].apply(lambda x: x > 0)
    trainer = Train(df,target ='amr_improve', model_type = model_type)
    if model_type == 'Random':
        trainer.train(max_f1=True)
        return
    else:
        xgb, report = trainer.train(max_f1 = True)


def train_ner_parser(model_type = 'XGBoost'):
    print('Start training NER_parser')
    ### NER
    pred_file = f"{google_pred_dir}/final_results_ner.csv"
    feature_file = good_dir / "ldc_ner_features_parser.csv"
    pred = pd.read_csv(pred_file)
    df = pd.read_csv(feature_file)
    different = False
    if 'id_y' in df.columns:
        df['id'] = df['id_y']
        df = df.drop(['id_x', 'id_y'], axis=1)
        different = True

    for col in df.columns:
        if '.1' in col or '.2' in col or '.3' in col or 'Unnamed' in col or 'helpfulness' in col:
            df.drop(col, axis=1, inplace=True)
            different = True
    if different:
        df.to_csv(feature_file, index=False)


    df_full = df.merge(pred[['id','helpfulness']], on = "id")
    df_full['did_amr_help'] = df_full['helpfulness']
    df = df_full.copy()
    df['amr_improve'] = df['did_amr_help'].apply(lambda x: x > 0)
    trainer = Train(df,target ='amr_improve', model_type = model_type)
    if model_type == 'Random':
        trainer.train(max_f1=True)
        return
    else:
        xgb, report = trainer.train(max_f1 = True)

def train_ner_gold(model_type = 'XGBoost'):
    print('Start training NER_gold')
    ### NER
    pred_file = f"{google_pred_dir}/final_results_ner_true.csv"
    feature_file = good_dir / "ldc_ner_features_true.csv"
    pred = pd.read_csv(pred_file)
    df = pd.read_csv(feature_file)
    different = False
    if 'id_y' in df.columns:
        df['id'] = df['id_y']
        df = df.drop(['id_x', 'id_y'], axis=1)
        different = True

    for col in df.columns:
        if '.1' in col or '.2' in col or '.3' in col or 'Unnamed' in col or 'helpfulness' in col:
            df.drop(col, axis=1, inplace=True)
            different = True
    if different:
        df.to_csv(feature_file, index=False)


    df_full = df.merge(pred[['id','helpfulness']], on = "id")
    df_full['did_amr_help'] = df_full['helpfulness']
    df = df_full.copy()
    df['amr_improve'] = df['did_amr_help'].apply(lambda x: x > 0)
    trainer = Train(df,target ='amr_improve', model_type = model_type)
    if model_type == 'Random':
        trainer.train(max_f1=True)
        return
    else:
        xgb, report = trainer.train(max_f1 = True)


def train_all(model_type = 'XGBoost'):
    print('Start training ALL')
    # pred_file = f"{google_pred_dir}/final_results_ner_true.csv"
    feature_file = data_dir / "all_data_features.csv"
    # pred = pd.read_csv(pred_file)
    df = pd.read_csv(feature_file)
    df['amr_improve'] = df['helpfulness'].apply(lambda x: int(x > 0))
    different = False
    if 'id_y' in df.columns:
        df['id'] = df['id_y']
        df = df.drop(['id_x', 'id_y'], axis=1)
        different = True

    for col in df.columns:
        if '.1' in col or '.2' in col or '.3' in col or 'Unnamed' in col:
            df.drop(col, axis=1, inplace=True)
            different = True
    if different:
        df.to_csv(feature_file, index=False)
    # for model in ['DecisionTree', 'RandomForest', 'XGBoost', 'Ensemble']:
    # for model in ['LinearRegression','Lasso','Ridge']:
    for model in ['LogisticRegression']:
        trainer = Train(df,target ='helpfulness', model_type = model)
        if model_type == 'Random':
            trainer.train(max_f1=True)
            return
        else:
            clf, report = trainer.train(max_f1 = True)



def train_random():
  func_list = [train_para, train_translation, train_LOGIC, train_pubmed, train_django]
  # func_list = [train_ldc_slang_parser, train_ldc_slang_gold, train_ner_parser, train_ner_gold, train_asilm]
  # func_list = [train_ner_parser, train_ner_gold]
  for func in func_list:
    func(model_type = "Random")



def main():
    # train_spider()
  # train_LOGIC()
  # train_para()
  # train_translation()
  # train_random()
  # train_django()
  # train_asilm()
  # train_gold_slang()
  # train_ldc_slang_parser()
  # train_ldc_slang_gold()
  # train_ner_parser()
  # train_ner_gold()
  # train_asilm()
  train_all()

if __name__ == '__main__':
  main()
###### Imprt libraries ######
import string

import matplotlib as plt
from collections import Counter

import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from spacy import displacy
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
from datasets import load_dataset
from datasets import concatenate_datasets
from textstat import textstat
from nltk import Tree
import nltk
from nltk.corpus import wordnet as wn
from mosestokenizer import *
from pathlib import Path
from amr_score import *
from get_embedding import *
# Download required resources
# nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("en_core_web_lg")
# nltk.download('large_grammars')
# nltk.download('punkt')
# nltk.download('wordnet')
from tct import TCT
import argparse
grammar = nltk.data.load('grammars/large_grammars/atis.cfg')
np.random.seed(0)
set_seed(0)
random.seed(0)

root_dir = Path(__file__).parent.parent.resolve()
data_dir = root_dir / "data"
feature_dir = data_dir / "featured"
to_process_dir = data_dir / "to_process"
tct_out_dir = data_dir / "tct_outputs"
google_pred_dir = r"~/Google Drive/My Drive/Zhijing&Yuen/amr_codes/data/predictions"



###### Class to get features ######
class PreProcess():
  def __init__(self, write_to_file):
      self.amrs = pd.read_csv(f'{google_pred_dir}/corrected_amrs.csv')
      self.write_to_file = write_to_file


###### Single-sentence features ######
  def tok(self, sentence):
    return word_tokenize(sentence)

  def count_punctuation(self, sentence):
      count = 0
      for char in sentence:
          if char in string.punctuation:
              count += 1
      return count

  def count_words_in_parentheses(self, sentence):
      # Check if there are parentheses in the sentence
      if '(' not in sentence or ')' not in sentence:
          return 0

      # Find all occurrences of text enclosed in parentheses
      matches = re.findall(r'\((.*?)\)', sentence)

      # Count the number of words in each occurrence
      count = 0
      for match in matches:
          words = match.split()
          count += len(words)

      return count

  def count_relative_pronouns(self, sentence):
      target_words = ['that', 'which', 'who', 'whom', 'whose','where','why']
      count = 0
      words = sentence.split()
      for word in words:
          if word.lower() in target_words:
              count += 1
      return count


  def count_words_in_quotations(self, sentence):
      # add all types of quotation, latex quotation ``
      # Check if there are single or double quotations in the sentence
      if "'" not in sentence and '"' not in sentence and "`" not in sentence:
          return 0

      # Find all occurrences of text enclosed in single or double quotations
      sentence = sentence.replace("``",""").replace("''",""")

      matches = re.findall(r'\'(.*?)\'|\"(.*?)\"', sentence)

      # Count the number of words in each occurrence
      count = 0
      for match in matches:
          words = match[0].split() if match[0] else match[1].split()
          count += len(words)

      return count

  def compute_bow(self, sentence):
      words = sentence.split(" ")
      words_list = [word.lower() for word in words if word not in string.punctuation]
      word_counts = Counter(words_list)
      return word_counts

  def entity_type_count(self, sentence):
      if type(sentence) != str:
          return 0
      doc = nlp(sentence)
      entity_types = [ent.label_ for ent in doc.ents]
      return len(set(entity_types))

  def unique_word_count(self, sentence):
      sentence = sentence.lower().translate(str.maketrans('', '', string.punctuation))
      return len(set(sentence.split()))

  def count_dependencies(self, sentence):
      # Parse the sentence using SpaCy
      if type(sentence) != str:
          return pd.Series(dtype = 'float64')
      doc = nlp(sentence)
      # Initialize a dictionary to store the counts of each dependency
      dep_counts = {}

      # Iterate through the tokens in the parsed sentence
      for token in doc:
          # Get the dependency label of the token
          dep_label = token.dep_

          # Update the count of the dependency label in the dictionary
          if dep_label in dep_counts:
              dep_counts[dep_label] += 1
          else:
              dep_counts[dep_label] = 1

      # Convert the dictionary to a pandas Series object
      return pd.Series(dep_counts)


  def count_pos(self, sentence):
      # Parse the sentence using SpaCy
      if type(sentence) != str:
          return pd.Series(dtype = 'float64')
      doc = nlp(sentence)

      # Initialize a dictionary to store the counts of each POS type
      pos_counts = {}

      # Iterate through the tokens in the parsed sentence
      for token in doc:
          # Get the POS type of the token
          pos_type = token.pos_

          # Update the count of the POS type in the dictionary
          if pos_type in pos_counts:
              pos_counts[pos_type] += 1
          else:
              pos_counts[pos_type] = 1

      # Convert the dictionary to a pandas Series object
      return pd.Series(pos_counts)

  def tok_format(self, tok):
    return f"{tok.orth_}/{tok.dep_}"

  def to_nltk_tree(self, node):
    if node.n_lefts + node.n_rights > 0:
      return Tree(self.tok_format(node), [self.to_nltk_tree(child) for child in node.children])
    else:
      return self.tok_format(node)

  def syntactic_complexity(self, sentence):
    # Remove leading special characters
    sentence = re.sub('^[^a-zA-Z0-9]*', '', sentence)
    if type(sentence) != str:
        return None
    doc = nlp(sentence)
    if len(doc) <= 1:
        return 0
    tree = self.to_nltk_tree(list(doc.sents)[0].root)
    if isinstance(tree,str):
      print(type(tree))
      print(sentence)
      height = None
    else:
      height = tree.height()
    return height


  def vocabulary_diversity(self, sentence):
    tokens = sentence.split()
    unique_tokens = set(tokens)
    type_token_ratio = len(unique_tokens) / len(tokens)
    return type_token_ratio



  def flesch_kincaid_grade_level(self, sentence):
    return textstat.flesch_kincaid_grade(sentence)

  def count_named_entities(self, sentence):
    # Remove leading special characters
    sentence = re.sub('^[^a-zA-Z0-9]*', '', sentence)
    if type(sentence) != str:
        return 0
    doc = nlp(sentence)
    named_entities = [ent for ent in doc.ents]
    return len(named_entities)


  def complex_sentence_structure(self, sentence):
    # Remove leading special characters
    sentence = re.sub('^[^a-zA-Z0-9]*', '', sentence)
    parse = nlp(sentence)
    return len(parse)


  def ambiguity(self,sentence):
    def count_meanings(sentence):
        words = word_tokenize(sentence)
        meanings = [len(wn.synsets(word)) for word in words]
        return sum(meanings)
    return count_meanings(sentence)

  def count_args(self,text):
    # Remove leading special characters
    text = re.sub('^[^a-zA-Z0-9]*', '', text)
    if type(text) != str:
        return 0
    doc = nlp(text)
    arguments = 0


    for token in doc:
        if token.dep_ in ("nsubj", "dobj", "iobj", "attr", "prep"):
            arguments += 1

    return arguments


  def count_adjuncts(self,text):
    # Remove leading special characters
    text = re.sub('^[^a-zA-Z0-9]*', '', text)
    if type(text) != str:
        return 0
    doc = nlp(text)
    adjuncts = 0

    for token in doc:
        if token.dep_ in ("advmod", "amod", "npadvmod", "nmod"):
            adjuncts += 1
    return adjuncts


  def count_negations(self, sentence):
    negation_words = [ "ain't", "aren't", "cannot", "can't", "couldn't", "daren't",
                      "didn't", "doesn't", "don't", "hadn't", "hardly", "hasn't", "haven't",
                      "isn't", "lack", "lacking", "lacks", "neither", "never", "no", "nobody",
                      "none", "nor", "not", "nothing", "nowhere", "oughtn't", "rarely",
                      "scarcely", "shan't", "shouldn't", "wasn't", "without", "won't", "wouldn't"]
    words = re.findall(r'\b\w+\b', sentence.lower())
    return sum(word in negation_words for word in words)


  def get_lcs_representation(self,sentence):
    # Tokenize and POS tag the sentence
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    # Identify the main verb and its arguments
    verb = None
    subject = None
    object = None
    indirect_object = None

    for i, (token, pos) in enumerate(pos_tags):
        if pos.startswith('VB'):  # Verb
            verb = token
        elif pos.startswith('NN'):  # Noun
            if subject is None:
                subject = token
            elif object is None:
                object = token
            elif indirect_object is None:
                indirect_object = token

    # Map the words to their corresponding primitive concepts
    def get_synset(word, pos):
        if word is None:
            return None
        pos_map = {
            'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n', 'IN': 'n',  # Nouns and prepositions/subordinating conjunctions
            'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',  # Verbs
            'JJ': 'a', 'JJR': 'a', 'JJS': 'a',  # Adjectives
            'RB': 'r', 'RBR': 'r', 'RBS': 'r', 'RP': 'r',  # Adverbs
        }
        synsets = wn.synsets(word.lower(), pos=pos_map.get(pos[:2], None))
        return synsets[0] if synsets else None

    verb_synset = get_synset(verb, 'VB')
    subject_synset = get_synset(subject, 'NN')
    object_synset = get_synset(object, 'NN')
    indirect_object_synset = get_synset(indirect_object, 'NN')

    # Construct the LCS representation
    lcs_representation = (verb_synset, subject_synset, object_synset, indirect_object_synset)
    return lcs_representation


  def lcs_primitive_concepts(self,sentence):
    lcs_representation = self.get_lcs_representation(sentence)
    return len([concept for concept in lcs_representation if concept is not None])


  def lcs_depth(self,sentence):
    lcs_representation = self.get_lcs_representation(sentence)
    depth = 0
    for concept in lcs_representation:
        if concept is not None:
            hypernym_paths = concept.hypernym_paths()
            max_path_length = max([len(path) for path in hypernym_paths])
            depth += max_path_length
    return depth


  def lcs_ambiguity(self,sentence):
      tokens = word_tokenize(sentence)
      pos_tags = nltk.pos_tag(tokens)
      pos_map = {
      'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n', 'IN': 'n',  # Nouns and prepositions/subordinating conjunctions
      'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',  # Verbs
      'JJ': 'a', 'JJR': 'a', 'JJS': 'a',  # Adjectives
      'RB': 'r', 'RBR': 'r', 'RBS': 'r', 'RP': 'r',  # Adverbs
  }

      ambiguity = 0
      for token, pos in pos_tags:
          pos_key = pos[:2]
          if pos_key in pos_map:
              synsets = wn.synsets(token, pos=pos_map[pos_key])
              ambiguity += len(synsets)

      return ambiguity

  def num_clauses(self,sentence):
    if type(sentence) != str:
        return 0
    doc = nlp(sentence)
    for token in doc:
      ancestors = [t.text for t in token.ancestors]
      children = [t.text for t in token.children]
      # print(token.text, "\t", token.i, "\t",
      #       token.pos_, "\t", token.dep_, "\t",
      #       ancestors, "\t", children)

    def find_root_of_sentence(doc):
      root_token = None
      for token in doc:
          if (token.dep_ == "ROOT"):
              root_token = token
      return root_token

    def find_other_verbs(doc, root_token):
      other_verbs = []
      for token in doc:
          ancestors = list(token.ancestors)
          if (token.pos_ == "VERB" and len(ancestors) == 1\
              and ancestors[0] == root_token):
              other_verbs.append(token)
      return other_verbs

    root_token = find_root_of_sentence(doc)
    other_verbs = find_other_verbs(doc, root_token)

    def get_clause_token_span_for_verb(verb, doc, all_verbs):
      first_token_index = len(doc)
      last_token_index = 0
      this_verb_children = list(verb.children)
      for child in this_verb_children:
          if (child not in all_verbs):
              if (child.i < first_token_index):
                  first_token_index = child.i
              if (child.i > last_token_index):
                  last_token_index = child.i
      return(first_token_index, last_token_index)

    token_spans = []
    all_verbs = [root_token] + other_verbs
    for other_verb in all_verbs:
        (first_token_index, last_token_index) = get_clause_token_span_for_verb(other_verb, doc, all_verbs)
        token_spans.append((first_token_index, last_token_index))

    sentence_clauses = []
    for token_span in token_spans:
      start = token_span[0]
      end = token_span[1]
      if (start < end):
        clause = doc[start:end]
        sentence_clauses.append(clause)
    sentence_clauses = sorted(sentence_clauses, key=lambda tup: tup[0])
    clauses_text = [clause.text for clause in sentence_clauses]

    return len(clauses_text)




###### Inter-sentence features ######
  def common_words(self, s1, s2):
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    return len(words1.intersection(words2))


  def word_order_similarity(self, s1, s2):
    words1 = s1.lower().split()
    words2 = s2.lower().split()
    common_words = set(words1).intersection(words2)

    if not common_words:
        return 0

    diff = sum(abs(words1.index(word) - words2.index(word)) for word in common_words)

    max_diff = sum(abs(len(words1) - 1 - 2 * i) for i in range(len(common_words)))

    if max_diff == 0:
        return 1

    normalized_diff = diff / max_diff
    similarity_score = 1 - normalized_diff

    return similarity_score


  def jaccard_similarity(self, sentence1, sentence2):
      set1 = set(sentence1.split())
      set2 = set(sentence2.split())
      intersection = set1.intersection(set2)
      union = set1.union(set2)
      return len(intersection) / len(union)


  def cosine_similarity_bow(self, s1, s2):
      bow1 = self.compute_bow(s1)
      bow2 = self.compute_bow(s2)
      # Create a combined set of keys from both BoW dictionaries
      combined_keys = set(bow1.keys()) | set(bow2.keys())

      # Create vectors for both BoW representations with the same dimensions
      vec1 = [bow1.get(key, 0) for key in combined_keys]
      vec2 = [bow2.get(key, 0) for key in combined_keys]

      # Calculate the cosine similarity
      cos_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
      return cos_sim


  def levenshtein_distance(self, s1, s2):
    if len(s1) < len(s2):
      s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
      distances_ = [i2+1]
      for i1, c1 in enumerate(s1):
          if c1 == c2:
              distances_.append(distances[i1])
          else:
              distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
      distances = distances_
    return distances[-1]


  def levenshtein_similarity_norm(self, s1, s2):
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_

    levenshtein_distance = distances[-1]
    max_distance = len(s1)

    dissimilarity_score = levenshtein_distance / max_distance
    similarity_score = 1 - dissimilarity_score

    return similarity_score


  def levenshtein_diff_norm(self, s1, s2):
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_

    levenshtein_distance = distances[-1]
    max_distance = len(s1)

    dissimilarity_score = levenshtein_distance / max_distance

    return dissimilarity_score


  def longest_common_subsequence(self, s1, s2):
      lengths = [[(0) for j in range(len(s2)+1)] for i in range(len(s1)+1)]
      for i, x in enumerate(s1):
          for j, y in enumerate(s2):
              if x == y:
                  lengths[i+1][j+1] = lengths[i][j] + 1
              else:
                  lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
      return lengths[-1][-1]


  def semantic_similarity(self, s1, s2):
      doc1 = nlp(s1)
      doc2 = nlp(s2)
      return doc1.similarity(doc2)


  def named_entity_overlap(self, s1, s2):
      doc1 = nlp(s1)
      doc2 = nlp(s2)
      entities1 = {ent.text.lower() for ent in doc1.ents}
      entities2 = {ent.text.lower() for ent in doc2.ents}
      return len(entities1.intersection(entities2))

  def structural_diff(self, sentence1, sentence2):
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)

    structure1 = [token.dep_ for token in doc1]
    structure2 = [token.dep_ for token in doc2]

    common_structure = set(structure1).intersection(structure2)
    diff = len(structure1) + len(structure2) - 2 * len(common_structure)

    return diff

  def synonym_diff(self, sentence1, sentence2, threshold=0.8):
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)


    def cosine_similarity(vec1, vec2):
      norm1 = np.linalg.norm(vec1)
      norm2 = np.linalg.norm(vec2)

      if norm1 == 0 or norm2 == 0:
          return 0

      return np.dot(vec1, vec2) / (norm1 * norm2)

    synonym_count = 0
    for token1 in doc1:
        for token2 in doc2:
            if cosine_similarity(token1.vector, token2.vector) >= threshold:
                synonym_count += 1

    return synonym_count


  def detail_diff(self,sentence1, sentence2):
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)

    details1 = [ent.text for ent in doc1.ents if ent.label_ in ["DATE", "TIME", "CARDINAL", "ORDINAL", "GPE"]]
    details2 = [ent.text for ent in doc2.ents if ent.label_ in ["DATE", "TIME", "CARDINAL", "ORDINAL", "GPE"]]

    common_details = set(details1).intersection(details2)
    diff = len(details1) + len(details2) - 2 * len(common_details)

    return diff

  def complex_word_count(self, sentence1, sentence2 = None, threshold=0.001):
    doc1 = nlp(sentence1)
    complex_words = 0
    for token in doc1:
        if token.prob < threshold:
            complex_words += 1

    if sentence2 is not None:
        doc2 = nlp(sentence2)
        for token in doc2:
            if token.prob < threshold:
                complex_words += 1

    return complex_words




  def lcs_similarity(self,sentence1, sentence2):
    lcs1 = self.get_lcs_representation(sentence1)
    lcs2 = self.get_lcs_representation(sentence2)

    similarities = []
    for concept1, concept2 in zip(lcs1, lcs2):
        if concept1 is not None and concept2 is not None:
            similarity = concept1.wup_similarity(concept2)
            if similarity is not None:
                similarities.append(similarity)

    if not similarities:
        return 0

    return sum(similarities) / len(similarities)


  def bow_diff(self,s1, s2):
    l1 = s1.split(" ")
    l2 = s2.split(" ")
    for i in range(0, len(l1)):
        for j in range(0, len(l2)):
            if l1[i] == l2[j]:
                l1[i] = 'damn'
                l2[j] = 'damn'
    l3 = []
    for item in l1:
        if item!='damn':
            l3.append(item)
    return len(l3)

  def bow_diff_perc(self, s1, s2):
    l1 = s1.split(" ")
    l2 = s2.split(" ")
    return self.bow_diff(s1,s2)/(0.5*(len(l1)+ len(l1)))






  def add_all_features(self, df, input_col1 = 'premise', input_col2 = 'hypothesis', save=True):
     inter_funcs = [
         self.bow_diff,
         self.bow_diff_perc,
         self.common_words,
         self.cosine_similarity_bow,
         self.detail_diff,
         self.jaccard_similarity,
         self.lcs_similarity,
         self.levenshtein_diff_norm,
         self.levenshtein_distance,
         self.levenshtein_similarity_norm,
         self.longest_common_subsequence,
         self.named_entity_overlap,
         self.semantic_similarity,
         self.structural_diff,
         self.synonym_diff,
         self.word_order_similarity]

     single_sent_funcs = [
         self.ambiguity,
         self.complex_sentence_structure,
         self.count_args,
         self.count_adjuncts,
         self.complex_word_count,
         self.count_named_entities,
         self.count_negations,
         self.count_punctuation,
         self.count_relative_pronouns,
         self.count_words_in_parentheses,
         self.count_words_in_quotations,
         self.flesch_kincaid_grade_level,
         self.lcs_ambiguity,
         self.lcs_depth,
         self.lcs_primitive_concepts,
         self.syntactic_complexity,
         self.vocabulary_diversity,
         self.num_clauses
            ]

     amr_col1 = f"{input_col1}_amr"
     amr_col2 = f"{input_col2}_amr"
     if 'id_y' in df.columns:
         df = df.rename(columns={'id_y': 'id'})


     # Apply the count_dependencies function to the input_col1 column and append the result to the DataFrame
     tqdm.pandas()
     pos_counts_pre = df[input_col1].progress_apply(self.count_pos)
     pos_counts_hyp = df[input_col2].progress_apply(self.count_pos)
     pos_counts_avg = (pos_counts_pre + pos_counts_hyp)/2

     # Add a prefix to the column names to distinguish between premise and hypothesis pos
     pos_counts_pre.columns = "pos_" + pos_counts_pre.columns + "_pre"
     pos_counts_hyp.columns = "pos_" + pos_counts_hyp.columns + "_hyp"
     pos_counts_avg.columns = "pos_" + pos_counts_avg.columns + "_avg"


     # Apply the count_dependencies function to the input_col1 column and append the result to the DataFrame
     dependency_counts_pre = df[input_col1].progress_apply(self.count_dependencies)
     dependency_counts_hyp = df[input_col2].progress_apply(self.count_dependencies)
     dependency_counts_avg = (dependency_counts_pre + dependency_counts_hyp)/2



     # Add a prefix to the column names to distinguish between premise and hypothesis dependencies
     dependency_counts_pre.columns = "dep_" + dependency_counts_pre.columns + "_pre"
     dependency_counts_hyp.columns = "dep_" + dependency_counts_hyp.columns + "_hyp"
     dependency_counts_avg.columns = "dep_" + dependency_counts_avg.columns + "_avg"



     # merge the original DataFrame with the dependency counts DataFrame
     df = pd.concat([df, dependency_counts_hyp, dependency_counts_pre, pos_counts_pre, pos_counts_hyp], axis=1)
     df = pd.concat([df, dependency_counts_avg, pos_counts_avg], axis=1)
     df.to_csv(self.write_to_file, index=False)
     print(df.shape, "after adding dependency counts and pos counts")


     # Get embeddings
     df = embed(df, input_col1, input_col2)
     df.fillna(df.mean(numeric_only=True), inplace=True)
     df.to_csv(self.write_to_file, index=False)
     print(df.shape, "after adding embeddings")

     for index, row in tqdm(df.iterrows()):
         row[input_col1] = str(row[input_col1])
         if isinstance(row[input_col2], float):
             print(row)
         row[input_col2] = str(row[input_col2])
         if row[input_col1].startswith("."):
             row[input_col1] = row[input_col1][1:].strip()
         if row[input_col2].startswith("."):
             row[input_col2] = row[input_col2][1:].strip()
         premise_tok = " ".join(self.tok(row[input_col1]))
         hypothesis_tok = " ".join(self.tok(row[input_col2]))
         premise_list = row[input_col1].split(" ")
         hypothesis_list = row[input_col2].split(" ")

         df.loc[index, 'string_len_pre'] = len(row[input_col1])
         df.loc[index, 'string_len_hyp'] = len(row[input_col2])
         df.loc[index, 'string_len_avg'] = (len(row[input_col1]) + len(row[input_col2])) / 2
         df.loc[index, "diff_string_len"] = abs(len(row[input_col1]) - len(row[input_col2]))

         word_list = premise_list
         pre_word_list = [word for word in word_list if word not in string.punctuation]
         df.loc[index, 'num_word_pre'] = len(word_list)

         word_list = hypothesis_list
         hyp_word_list = [word for word in word_list if word not in string.punctuation]
         df.loc[index, 'num_word_hyp'] = len(word_list)
         df.loc[index, 'num_word_avg'] = (len(premise_list) + len(hypothesis_list)) / 2

         df.loc[index, 'numerical_pre'] = sum(c.isdigit() for c in row[input_col1])
         df.loc[index, 'numerical_hyp'] = sum(c.isdigit() for c in row[input_col2])
         df.loc[index, 'numerical_avg'] = (sum(c.isdigit() for c in row[input_col1]) + sum(
             c.isdigit() for c in row[input_col2])) / 2

         if 'ldc_slang' in row['id']:
             temp_amr = pd.read_csv(data_dir / 'ldc_slang_hand.csv')
             df['id'] = temp_amr['id']
             df.loc[index, amr_col1] = temp_amr[temp_amr['id'] == row['id']]['true_premise_amr'].values[0]
             df.loc[index, amr_col2] = temp_amr[temp_amr['id'] == row['id']]['hand_hypothesis_amr'].values[0]
         elif 'ldc_dev' in row['id']:
             df.loc[index, amr_col1] = \
             self.amrs[self.amrs['id'] == f"{'_'.join(row['id'].split('_')[:-1])}_p"]['amr'].values[0]
             df.loc[index, amr_col2] = self.amrs[self.amrs['id'] == row['id']]['amr'].values[0]
         elif 'newstest' in row['id']:
             df.loc[index, amr_col1] = self.amrs[self.amrs['id'] == f"{row['id']}_en"]['amr'].values[0]
             df.loc[index, amr_col2] = self.amrs[self.amrs['id'] == f"{row['id']}"]['amr'].values[0]
         else:
             print(row['id'])
             df.loc[index, amr_col1] = self.amrs[self.amrs['id'] == f"{row['id']}"]['amr'].values[0]
             df.loc[index, amr_col2] = self.amrs[self.amrs['id'] == f"{row['id']}"]['amr'].values[0]

         for func in inter_funcs:
             func_name = func.__name__
             df.loc[index, f"{func_name}"] = func(premise_tok, hypothesis_tok)

         for func in single_sent_funcs:
             func_name = func.__name__
             # print(index, f"{func_name}_pre")
             df.loc[index, f"{func_name}_pre"] = func(premise_tok)
             df.loc[index, f"{func_name}_hyp"] = func(hypothesis_tok)
             df.loc[index, f"{func_name}_avg"] = (func(premise_tok) + func(hypothesis_tok)) / 2

         if save and index % 100 == 0:
             df.to_csv(self.write_to_file, index=False)
             if index % 2000 == 0:
                 print(df.shape, 'rows saved at index',index, flush=True)
     df.to_csv(self.write_to_file, index=False)
     print(df.shape, 'rows saved', flush=True)

     # Get AMR features
     df = get_amr_features_two_sent(df, amr_col1=amr_col1, amr_col2=amr_col2)
     df.to_csv(self.write_to_file, index=False)
     df = get_3_amr_features(df, amr_pred=amr_col1, amr_gold=amr_col2)
     df.to_csv(self.write_to_file, index=False)
     print(df.shape, "after adding amr features")
     return df


  def get_features_one_sent(self,df, input_col, amr_col  = None, save=True):
    if amr_col is None:
        amr_col = f'{input_col}_amr'
    else :
        amr_col = amr_col
    single_sent_funcs = [
      self.ambiguity,
      self.complex_sentence_structure,
      self.count_args,
      self.count_adjuncts,
      self.complex_word_count,
      self.count_named_entities,
      self.count_negations,
      self.count_punctuation,
      self.count_relative_pronouns,
      self.count_words_in_parentheses,
      self.count_words_in_quotations,
      self.flesch_kincaid_grade_level,
      self.lcs_ambiguity,
      self.lcs_depth,
      self.lcs_primitive_concepts,
      self.syntactic_complexity,
      self.vocabulary_diversity,
      self.num_clauses
      ]



    tqdm.pandas()
    df['string_len'] = df[input_col].progress_apply(lambda x: len(x))
    df['num_word'] = df[input_col].progress_apply(lambda x: len(x.split()))
    # Apply the count_dependencies function to the input_col1 column and append the result to the DataFrame
    pos_counts = df[input_col].progress_apply(self.count_pos)
    # Add a prefix to the column names to distinguish between premise and hypothesis pos
    pos_counts.columns = "pos_" + pos_counts.columns
    # Apply the count_dependencies function to the input_col1 column and append the result to the DataFrame
    dependency_counts = df[input_col].progress_apply(self.count_dependencies)
    # Add a prefix to the column names to distinguish between premise and hypothesis dependencies
    dependency_counts.columns = "dep_" + dependency_counts.columns
    # merge the original DataFrame with the dependency counts DataFrame
    df = pd.concat([df, dependency_counts, pos_counts], axis=1)
    df.to_csv(self.write_to_file, index=False)
    print(df.shape, 'rows saved after adding dep_ and pos_ counts', flush=True)


    for index, row in tqdm(df.iterrows()):
      tok = " ".join(self.tok(row[input_col]))

      for func in single_sent_funcs:
        func_name = func.__name__
        # print(index, f"{func_name}_pre")
        df.loc[index, f"{func_name}"] = func(tok)

        if 'en' in df.columns:
            df.loc[index, amr_col] = self.amrs[self.amrs['id'] == f"{row['id']}_en"]['amr'].values[0]
        elif 'ldc_entity' in row['id']:
            pass
        else:
            df.loc[index, amr_col] = self.amrs[self.amrs['id'] == f"{row['id']}"]['amr'].values[0]

        if save and index % 100 == 0:
            df.to_csv(self.write_to_file, index=False)

            # Get AMR features
            df = get_amr_features_one_sent(df, amr_col=amr_col)
            df.fillna(df.mean(numeric_only=True), inplace=True)
            df.to_csv(self.write_to_file, index=False)
            print(df.shape, 'rows saved after adding AMR features', flush=True)
    df = get_amr_features_one_sent(df, amr_col=amr_col)
    print(df.shape, 'rows saved after adding AMR features', flush=True)
    df.to_csv(self.write_to_file, index=False)
    return df






##### Get features for PAWS #####
class PAWS_preprocessor(PreProcess):
    def __init__(self,write_to_file):
        super().__init__(write_to_file)

    def get_features(self,df):
        print(self.write_to_file)
        self.add_all_features(df,input_col1='premise',input_col2='hypothesis')
        return df


##### Get features for WMT #####
class WMT_preprocessor(PreProcess):
    def __init__(self,write_to_file):
        super().__init__(write_to_file)
        self.write_to_file = data_dir / 'wmt_text_features.csv'
        self.write_to_file = data_dir / 'wmt_text_features.csv'
    def get_features(self,df):
        df = self.get_features_one_sent(df,input_col='en')
        input_col1 = 'en'
        input_col2 = 'de'
        for index, row in tqdm(df.iterrows()):
            # row['premise'] = str(row['premise'])
            if isinstance(row[input_col2], float):
                print(row)
                # row['hypothesis'] = str(row['hypothesis'])
            if row[input_col1].startswith("."):
                row[input_col1] = row[input_col1][1:].strip()
            if row[input_col2].startswith("."):
                row[input_col2] = row[input_col2][1:].strip()
            premise_tok = " ".join(self.tok(row[input_col1]))
            hypothesis_tok = " ".join(self.tok(row[input_col2]))
            premise_list = row[input_col1].split(" ")
            hypothesis_list = row[input_col2].split(" ")

            df.loc[index, 'string_len_en'] = len(row[input_col1])
            df.loc[index, 'string_len_de'] = len(row[input_col2])
            df.loc[index, 'string_len_avg'] = (len(row[input_col1]) + len(row[input_col2])) / 2
            df.loc[index, "diff_string_len"] = abs(len(row[input_col1]) - len(row[input_col2]))

            word_list = premise_list
            pre_word_list = [word for word in word_list if word not in string.punctuation]
            df.loc[index, 'num_word_en'] = len(word_list)

            word_list = hypothesis_list
            hyp_word_list = [word for word in word_list if word not in string.punctuation]
            df.loc[index, 'num_word_de'] = len(word_list)
            df.loc[index, 'num_word_avg'] = (len(premise_list) + len(hypothesis_list)) / 2

        return df



##### Get features for WMT #####
class LOGIC_preprocessor(PreProcess):
    def __init__(self,write_to_file = feature_dir / 'logic_text_features.csv'):
        super().__init__(write_to_file)


    def get_features(self, df, save=True):
        df = self.get_features_one_sent(df,input_col='source_article',save=save)
        return df


class PUBMED45_preprocessor(PreProcess):
    def __init__(self,write_to_file = feature_dir / 'pubmed45_text_features.csv'):
        super().__init__(write_to_file)
    def interaction_len(self, inter_lst):
        return len(inter_lst)

    def count_interactions(self, sentence):
        text = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(text)
        verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
        return len(verbs)

    def average_entity_position(self, sentence, interaction_list):
        words = sentence.split()
        positions = [i for i, word in enumerate(words) if word in interaction_list]
        return sum(positions) / len(positions) if positions else 0

    def average_word_order(self, sentence, interaction_list):
        words = sentence.split()
        positions = [i for i, word in enumerate(words) if word in interaction_list]
        return sum(positions) / (len(words) * len(positions)) if positions else 0

    def interaction_tfidf(self, sentence, interaction_list):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([sentence])  # Put sentence in a list
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()

        tfidf_dict = dict(zip(feature_names, tfidf_scores.mean(axis=0)))

        interaction_tfidf = {word: tfidf_dict.get(word, 0) for word in interaction_list}

        return interaction_tfidf

    def interation_distance(self, sentence, interaction_list):
        words = sentence.split()
        positions = [i for i, word in enumerate(words) if word in interaction_list]

        if len(positions) < 2:
            return 0

        distances = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]

        return sum(distances) / len(distances)

    def cooccurrence_count(self, sentence, interaction_list):
        return sum(1 for word in interaction_list if word.lower() in sentence.lower())

    def get_features(self, df, save=True):
        self.get_features_one_sent(df,input_col='sentence', save=save)
        if save:
            df.to_csv(self.write_to_file, index = False)
        df['interaction_len'] = df['interaction'].apply(self.interaction_len)
        df['interaction_tfidf'] = df.apply(lambda x: self.interaction_tfidf(x['sentence'], x['interaction']), axis=1)
        df['interaction_distance'] = df.apply(lambda x: self.interation_distance(x['sentence'], x['interaction']), axis=1)
        df['cooccurrence_count'] = df.apply(lambda x: self.cooccurrence_count(x['sentence'], x['interaction']), axis=1)
        if save:
            df.to_csv(self.write_to_file, index = False)
        return df



class DJANGO_preprocessor(PreProcess):
    def __init__(self, write_to_file = feature_dir / 'django_text_features.csv'):
        super().__init__(write_to_file)

    def get_features(self, df, save=True):
        df = self.get_features_one_sent(df,input_col='text', save=save)
        return df

class LDC_NER_preprocessor(PreProcess):
    def __init__(self, write_to_file = feature_dir / 'ldc_ner_text_features.csv'):
        super().__init__(write_to_file)

    def get_features(self, df, save=True):
        df = self.get_features_one_sent(df,input_col='text',amr_col = 'true_amr', save=save)
        return df


class SPIDER_preprocessor(PreProcess):
    def __init__(self, write_to_file = feature_dir / 'spider_text_features.csv'):
        super().__init__(write_to_file)

    def count_select(self, sentence):
        # count the number of "SELECT" in the sentence
        return sentence.count("SELECT")


    #### Count the number of SELECT, WHERE,GROUP BY, HAVING, ORDER BY, LIMIT,JOIN, INTERSECT, EXCEPT, UNION, NOT IN, OR, AND, EXISTS, LIKE in the sentence
    def count_select(self, sentence):
        # count the number of "SELECT" in the sentence
        return sentence.count("SELECT")

    def count_where(self, sentence):
        # count the number of "WHERE" in the sentence
        return sentence.count("WHERE")

    def count_group_by(self, sentence):
        # count the number of "GROUP BY" in the sentence
        return sentence.count("GROUP BY")

    def count_having(self, sentence):
        # count the number of "HAVING" in the sentence
        return sentence.count("HAVING")

    def count_order_by(self, sentence):
        # count the number of "ORDER BY" in the sentence
        return sentence.count("ORDER BY")

    def count_limit(self, sentence):
        # count the number of "LIMIT" in the sentence
        return sentence.count("LIMIT")

    def count_join(self, sentence):
        # count the number of "JOIN" in the sentence
        return sentence.count("JOIN")

    def count_intersect(self, sentence):
        # count the number of "INTERSECT" in the sentence
        return sentence.count("INTERSECT")

    def count_except(self, sentence):
        # count the number of "EXCEPT" in the sentence
        return sentence.count("EXCEPT")

    def count_union(self, sentence):
        # count the number of "UNION" in the sentence
        return sentence.count("UNION")

    def count_not_in(self, sentence):
        # count the number of "NOT IN" in the sentence
        return sentence.count("NOT IN")

    def count_or(self, sentence):
        # count the number of "OR" in the sentence
        return sentence.count("OR")

    def count_and(self, sentence):
        # count the number of "AND" in the sentence
        return sentence.count("AND")

    def count_exists(self, sentence):
        # count the number of "EXISTS" in the sentence
        return sentence.count("EXISTS")

    def count_like(self, sentence):
        # count the number of "LIKE" in the sentence
        return sentence.count("LIKE")

    def count_keywords(self, sentence):
        # count the number of keywords in the sentence
        return self.count_select(sentence) + \
                self.count_where(sentence) + \
                self.count_group_by(sentence) + \
                self.count_having(sentence) + \
                self.count_order_by(sentence) + \
                self.count_limit(sentence) + \
                self.count_join(sentence) + \
                self.count_intersect(sentence) + \
                self.count_except(sentence) + \
                self.count_union(sentence) + \
                self.count_not_in(sentence) + \
                self.count_or(sentence) + \
                self.count_and(sentence) + \
                self.count_exists(sentence) + \
                self.count_like(sentence)


    def get_features(self, df, save=True):
        if 'ground_truth' not in df.columns and 'groundtruth' in df.columns:
            df['ground_truth'] = df['groundtruth']
        df = self.get_features_one_sent(df,input_col='question', save=save)
        func_list = [self.count_select, self.count_where, self.count_group_by, self.count_having, self.count_order_by,
                     self.count_limit, self.count_join, self.count_intersect, self.count_except, self.count_union,
                     self.count_not_in, self.count_or, self.count_and, self.count_exists, self.count_like]
        for func in func_list:
            df[func.__name__] = df.apply(lambda x: func(x['ground_truth']), axis=1)
        df['count_keywords'] = df.apply(lambda x: self.count_keywords(x['ground_truth']), axis=1)

        return df



def main(args):
    input_file = args.data_file
    dataset = args.dataset
    output_file = args.output_file
    df = pd.read_csv(input_file)
    if dataset in ['paws','ldc_slang','ldc_slang_gold','asilm','ldc_dev']:
        processor = PAWS_preprocessor(write_to_file=output_file)
    elif dataset in ['django']:
        processor = DJANGO_preprocessor(write_to_file=output_file)
    elif dataset in ['logic']:
        processor = LOGIC_preprocessor(write_to_file=output_file)

    elif dataset in ['spider']:
        processor = SPIDER_preprocessor(write_to_file=output_file)
    elif dataset in ['entity_recog_gold','entity_recog']:
        processor = LDC_NER_preprocessor(write_to_file=output_file)
    elif dataset in ['newstest','wmt']:
        processor = WMT_preprocessor(write_to_file=output_file)
    elif dataset in ['pubmed', 'pubmed45']:
        processor = PUBMED45_preprocessor(write_to_file=output_file)


    df = processor.get_features(df)
    df.to_csv(output_file, index=False)

    #### Get TCT Features
    tct_processor = TCT(output_file)
    with_tct = tct_processor.get_tct()
    with_tct.to_csv(output_file, index=False)








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute linguist features')
    parser.add_argument('--data_file', type=str, default=data_dir/"classifier_inputs/updated_data_input - classifier_input.csv", help='the csv file to process')
    parser.add_argument('--dataset', type=str, default='logic', help='the dataset name')
    parser.add_argument('--output_file', type=str, default = data_dir/'featured', help='whether to save the features')

    args = parser.parse_args()
    main(args)
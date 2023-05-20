from efficiency.log import show_var


class Data:
    data_names = 'esnli negation ' \
                 'amazon_counterfactual_en paws proofwriter'.split()
    data_names = ['paws',]
    # 'boolean comparative conditional counting winowhy aqua gsm8k ' \

    model_version2engine = {
        'gpt4': "gpt-4",
        'gpt3.5': "gpt-3.5-turbo",
        'gpt3.3': "text-davinci-003",
        'gpt3.2': "text-davinci-002",
        # 'gpt3.5d3': "text-davinci-003",
        # 'gpt3.5d2': "text-davinci-002",
        # 'gpt3': "davinci", # all these and below cannot really answer the AMR question in zero shot
        # 'gpt3c': "text-curie-001",
        # 'gpt3b': "text-babbage-001",
        # 'gpt3a': "text-ada-001",
        # 'gpt3d': "text-davinci-001",
    }

    # data_name = data_names[-1]

    # data_name = 'comparative'
    root = '/home/zjin/proj'

    # root = '..'

    def preprocess_to_random_order(self):
        # Get raw data
        from efficiency.log import fread
        data_paths = [
            f'{self.root}/2208_optr/data/finetune/datasets/{self.data_name}/test.jsonl',
            f'{self.root}/2208_optr/data/test/{self.data_name}.jsonl',  # TODO: move files to a local path
            f'{self.root}/2208_optr/data/test_2023/{self.data_name}.json',
            f'{self.root}/2208_optr/data/test_2023/{self.data_name}.jsonl',
            f'{self.root}/2208_optr/data/finetune/datasets/{self.data_name}/train.jsonl',
        ]

        for data_path in data_paths:
            data = fread(data_path, verbose=True)
            if data:
                self.data_path = data_path
                break
        try:
            self.data_path
        except:
            print('[Error] No valid file path has been found among: ', data_paths)
            import pdb;
            pdb.set_trace()

        show_var(['len(data)'])
        if not len(data):
            import pdb;
            pdb.set_trace()
        # Starts preprocessing
        if 'rows' in data:
            data = data['rows']
            data = [i['row'] for i in data]
            if 'sentence1' in data[0]:
                data = [{'premise': i['sentence1'], 'hypothesis': i['sentence2'], 'truth': i['label']}
                        for i in data]
                '''
                PAWS data
                "row": {
                    "id": 1,
                    "sentence1": "This was a series of nested angular standards , so that measurements in azimuth and elevation could be done directly in polar coordinates relative to the ecliptic .",
                    "sentence2": "This was a series of nested polar scales , so that measurements in azimuth and elevation could be performed directly in angular coordinates relative to the ecliptic .",
                    "label": 0
                  },
                '''
            else:
                data = [{'premise': i['text'], 'truth': i['label_text']}
                        for i in data]
                '''
                Amazon Counterfactual
                "row": {
                    "text": "If you want great sound, buy a headset or a $1000 system.",
                    "label": 1,
                    "label_text": "counterfactual"
                },
                '''

        elif 'src' in data[0]:
            for datum_i, datum in enumerate(data):
                text = datum['src'].rsplit('Premise: ', 1)[-1].split("\nLet's think step by step: The answer is", 1)[0]
                prem, hyp = text.split('\nHypothesis: ')
                new_datum = {'premise': prem, "hypothesis": hyp, 'truth': datum['lbl'], }
                data[datum_i] = new_datum
        elif 'sentence2' in data[0]:
            data = [{'premise': i['sentence1'], 'hypothesis': i['sentence2'], 'truth': i['label']}
                    for i in data]
        else:
            for datum_i, datum in enumerate(data):
                datum['truth'] = datum['answer']
                del (datum['answer'])


        from efficiency.function import set_seed
        set_seed()
        import random
        random.shuffle(data)
        import json
        from efficiency.log import fwrite
        writeout = '\n'.join([json.dumps(i) for i in data])
        fwrite(writeout, self.preprocessed_file, verbose=True)

    def __init__(self, data_name, model_version='gpt4', test_cot=False, fix_error=False, num=None):
        self.data_name = data_name
        self.model_version = model_version
        self.test_cot = test_cot

        self.tqdm_desc = f'Model={self.model_version} ({self.test_cot}), Data={self.data_name}'

        self.preprocessed_file = f'{self.root}/2208_optr/data/test_2023/random_{self.data_name}.jsonl'

        import os
        from efficiency.log import fread

        if not os.path.exists(self.preprocessed_file):
            self.preprocess_to_random_order()
        data = fread(self.preprocessed_file, verbose=False)

        if not len(data):
            import pdb;
            pdb.set_trace()
            self.preprocessed_file

        self.data = data[:num]
        cot_str = '_cot' if test_cot else ''
        suffix = '_error' if fix_error else ''
        self.output_file = f'data/outputs/{self.model_version}{cot_str}_{self.data_name}{suffix}.csv'
        self.gpt_cache_file = f'data/outputs/cache/{self.model_version}_query_n_pred.csv'
        self.prev_file = f'data/outputs/{self.model_version}_{self.data_name}.csv'
        self.data_existing = fread(self.output_file)

        if fix_error:
            from efficiency.log import fread
            tricky_direct_queries = fread(self.prev_file, verbose=False)
            key = [i for i in tricky_direct_queries[0].keys() if i.startswith('Avg:')][0]
            self.tricky_data = [row for i, row in enumerate(tricky_direct_queries) if row[key] == 0]
            if self.data_name == 'paws':
                self.tricky_data = self.tricky_data[:-3]


class Tester:
    default_choices = ["Yes",
                       "Could be yes with certain assumptions",
                       "Can never be yes regardless of any possible additional information",
                       ]
    alternative_choices = ["Yes", "No"]
    default_query_suffix = '''Begin your answer with "Yes," "Could be yes with certain assumptions," or "Can never be yes regardless of any possible additional information," and then provide further reasoning or evidence in your explanation.'''
    data_name2query_suffix_dict = {
        'winowhy': 'Both are likely, so which one fits common sense better? Just answer the most likely phrase.'
    }
    default_amr_relation = 'can we formally deduce the second statement from the first statement'
    data_name2amr_relation_dict = {
        'paws': 'can we conclude whether the two sentences are paraphrases of each other'
    }
    cot2default_query = {'direct': 'Given "{prem}", can we formally deduce that "{hyp}"?',
                         'cot': [
                             "What are the commonalities and differences between the two AMRs?",
                             'Based on the information above, {amr_relation}?',
                         ],
                         }
    data_name_n_cot2query_dict = {
        'amazon_counterfactual_en': {
            'cot': ['From the AMR, can we see whether there is a counterfactual expression in the sentence?'],
            'direct': 'Is there a counterfactual expression in the sentence "{prem}"?',
        },
        'paws': {
            'direct': 'Are the following two sentences paraphrases of each other?\n- {prem}\n- {hyp}',
        },
        # 'esnli':f'Given "{prem}", without commonsense knowledge and just limiting to the literal meaning, ' \
        #             #             f'can we infer that "{hyp}"?'

    }

    def __init__(self, default_max_tokens=1):
        choices = self.alternative_choices if D.data_name in {'paws', 'amazon_counterfactual_en'} \
            else self.default_choices
        from efficiency.log import verbalize_list_of_options
        choices = verbalize_list_of_options(choices)
        default_query_suffix = \
            f'Begin your answer with {choices}, and then provide further reasoning or evidence in your explanation.'
        data_name2query_suffix = lambda i: self.data_name2query_suffix_dict.get(i, default_query_suffix)
        data_name2amr_relation = lambda i: self.data_name2amr_relation_dict.get(i, self.default_amr_relation)

        self.max_tokens = 1000 if (D.data_name in {'proofwriter'}) or D.test_cot else default_max_tokens

        self.query_suffix = data_name2query_suffix(D.data_name)
        self.amr_relation = data_name2amr_relation(D.data_name)

    def data_name_n_cot2query(self):
        cot_str = 'cot' if D.test_cot else 'direct'
        if D.data_name in self.data_name_n_cot2query_dict:
            cot2query_dict = self.data_name_n_cot2query_dict[D.data_name]
            if cot_str in cot2query_dict:
                return cot2query_dict[cot_str]
        return self.cot2default_query[cot_str]

    def run_gpt4_cot(self, test_tricky_ones=False):
        data = D.data
        import pandas as pd
        from tqdm import tqdm

        if test_tricky_ones:
            data = D.tricky_data
        show_var(['len(data)', 'data[0]'])

        existing_n = len(D.data_existing)
        data[:existing_n] = D.data_existing
        for datum_i, datum in tqdm(list(enumerate(data)), desc=D.tqdm_desc):
            if datum_i < existing_n:
                continue

            prem = datum['premise']
            gold = datum['truth']

            all_sentences = [datum[i] for i in ['premise', 'hypothesis'] if i in datum]
            queries = [f'What is the Abstract Meaning Representation (AMR) for the text "{i}"?'
                       for i in all_sentences]
            queries = [f'What is the Abstract Meaning Representation (AMR) for the sentence "{i}"?'
                       for i in all_sentences]

            queries += self.data_name_n_cot2query()
            queries[-1] += '\n' + self.query_suffix
            queries = [i.replace('{amr_relation}', self.amr_relation) for i in queries]

            for query_i, query in enumerate(queries):
                max_tokens = self.max_tokens
                if query.endswith(', and then provide further reasoning or evidence in your explanation.'):
                    max_tokens = 1
                response = chat.ask(
                    query, continued_questions=query_i, enable_pdb=False,
                    turn_off_cache=(not query.startswith('What is the Abstract Meaning Representation')),
                    max_tokens=max_tokens)

                datum[f'query{query_i}'] = query
                datum[f'pred{query_i}'] = response

            if 'explanation' in datum:
                # move it to the last
                expl = datum['explanation']
                del (datum['explanation'])
                datum['explanation'] = expl
            df = pd.DataFrame(data[:datum_i + 1])
            df.to_csv(D.output_file, index=False)

    def run_gpt4(self):
        data = D.data
        import pandas as pd
        from tqdm import tqdm

        show_var(['len(data)', 'data[0]'])

        for datum_i, datum in tqdm(list(enumerate(data)), desc=D.tqdm_desc):

            prem = datum['premise']
            gold = datum['truth']

            query = self.data_name_n_cot2query()
            if '{hyp}' in query:
                hyp = datum['hypothesis']
                query = query.replace('{hyp}', hyp)
            query = query.replace('{prem}', prem)
            query += '\n' + self.query_suffix

            response = chat.ask(query, max_tokens=self.max_tokens, enable_pdb=False)

            datum['pred'] = response
            datum['query'] = query
            if 'explanation' in datum:
                # move it to the last
                expl = datum['explanation']
                del (datum['explanation'])
                datum['explanation'] = expl

            df = pd.DataFrame(data[:datum_i + 1])
            df.to_csv(D.output_file, index=False)


class Scorer:
    gold_label2binary = {
        'yes': 1,
        'entailment': 1,
        'counterfactual': 1,
        'neutral': 0.5,
        'unknown': 0.5,
        'contradiction': 0,
        'not-counterfactual': 0,
        'no': 0,
    }
    prefix2binary = {
        'Yes': 1,
        'Could be yes with certain assumptions': 0.5,
        'Can never be yes regardless of any possible additional information': 0,
        'No': 0,
    }

    @staticmethod
    def check_if_doable():
        import os
        return os.path.exists(D.output_file)

    def get_df(self):
        import pandas as pd
        df = pd.read_csv(D.output_file)
        return df

    def score_results(self):
        df = self.get_df()
        df = self.preprocess_df(df)
        df['score'] = df['pred_binary'] == df['truth_binary']
        report = {
            'score': round(df['score'].mean() * 100, 2),
            '# error examples': df['score'].value_counts().get(False, 0),
        }
        report.update({
            '# examples': f"{report['# error examples']}/{len(df)}"
        })
        report_str = f"Performance={report['score']:.2f}%\t({report['# examples']})\t File={D.output_file}\t"
        print()
        print(report_str)

        # import pdb;pdb.set_trace()
        # df[['truth', 'truth_binary', 'pred', 'pred_binary']]
        return report

    def preprocess_df(self, df):
        def convert_to_binary(value):
            for prefix, binary in self.prefix2binary.items():
                if value.startswith(prefix):
                    return binary
            return None

        pred_key = [i for i in df.columns if i.startswith('pred')][-1]  # for cot, we use the last pred{i} in the df
        df['pred_binary'] = df[pred_key].apply(convert_to_binary)

        def convert_truth_to_binary(value):
            return self.gold_label2binary.get(value.lower() if isinstance(value, str) else value, value)  # Return None
            # if the value is not found in the
            # dictionary

        df['truth_binary'] = df['truth'].apply(convert_truth_to_binary)
        return df




def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-eval', action='store_true', help='Enable eval mode')
    args = parser.parse_args()

    return args

'''
python code/gpt4.py -eval
python code/gpt4.py -eval
'''

if __name__ == '__main__':
    from efficiency.function import set_seed

    set_seed()
    '''
    experiment log: proofwriter shows error
    '''
    args = get_args()
    eval_mode = args.eval

    if eval_mode:
        list_test_cot = [False, True]
        list_model_version = list(Data.model_version2engine)
        list_data_name = Data.data_names
    else:
        # Training mode # TODO: make them as args
        list_test_cot = [False, True][1:]
        list_model_version = list(Data.model_version2engine)[:1]
        list_data_name = Data.data_names[::-1]

    from itertools import product

    combs = list(product(list_test_cot, list_model_version, list_data_name))
    print(combs)
    if not eval_mode: import pdb;pdb.set_trace()
    all_reports = []
    from efficiency.nlp import Chatbot


    for test_cot, model_version, data_name in combs:
        D = Data(data_name, model_version=model_version, test_cot=test_cot)
        chat = Chatbot(output_file=D.gpt_cache_file, cache_files=[D.prev_file, D.output_file])

        if eval_mode:
            try:
                if not Scorer.check_if_doable():
                    continue
                report = Scorer().score_results()
                report.update({
                    'model_version': model_version,
                    'cot': test_cot,
                    'data_name': data_name,
                })
                all_reports.append(report)
            except:
                pass
            continue

        if test_cot:
            tester = Tester().run_gpt4_cot()
        else:
            tester = Tester().run_gpt4()
    import pandas as pd

    df = pd.DataFrame(all_reports, index=None)
    # Create a pivot table with the difference in scores
    pivot_table = df.pivot_table(index=['data_name', 'model_version', ], columns='cot', values='score')
    pivot_table['score_diff'] = pivot_table[True] - pivot_table[False]
    print(pivot_table)

    import pdb;

    pdb.set_trace()
    pivot_table = pivot_table.reset_index().drop([False, True], axis=1)

    import pdb;

    pdb.set_trace()

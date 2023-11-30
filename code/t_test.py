import pandas as pd
from scipy import stats
from pathlib import Path
import os

root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
output_dir = data_dir / "output_gpt4"


amr_gold = pd.read_csv(output_dir / 'requests_amr_entity_recog_gold.csv')
amr_parser = pd.read_csv(output_dir / 'requests_amr_entity_recog.csv')
direct = pd.read_csv(output_dir / 'requests_direct_entity_recog.csv')

amr_gold = amr_gold[['text', 'f1']]
direct = direct[['text', 'f1']]



# Run t-test on the f1 scores between amr_gold and direct
amr_gold_f1 = amr_gold['f1']
direct_f1 = direct['f1']
t, p = stats.ttest_ind(amr_gold_f1, direct_f1)
ci = stats.ttest_ind(amr_gold_f1, direct_f1).confidence_interval()
print('amr_gold vs. direct')
print('Difference in mean f1 scores between amr_gold and direct')
print(amr_gold_f1.mean() - direct_f1.mean())
print(f't = {t}, p = {p}')

print("95 % confidence interval for the difference between amr_gold and direct")
print(ci)




# Run t-test on the f1 scores between amr_parser and direct

amr_parser_f1 = amr_parser['f1']
t, p = stats.ttest_ind(amr_parser_f1, direct_f1)
ci = stats.ttest_ind(amr_parser_f1, direct_f1).confidence_interval()
print('amr_parser vs. direct')
print('Difference in mean f1 scores between amr_parser and direct')
print(amr_parser_f1.mean() - direct_f1.mean())

print(f't = {t}, p = {p}')

print("95 % confidence interval for the difference between amr_parser and direct")
print(ci)

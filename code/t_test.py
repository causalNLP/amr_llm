import pandas as pd
from scipy import stats
from pathlib import Path
import os

root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
output_dir = data_dir / "output_gpt4"
#
# for dataset in ['paws','newstest','logic','pubmed']:
#     amr = pd.read_csv(output_dir / f'requests_amr_{dataset}.csv')
#     direct = pd.read_csv(output_dir / f'requests_direct_{dataset}.csv')
#
#
# amr_gold = pd.read_csv(output_dir / 'requests_amr_entity_recog_gold.csv')
# amr_parser = pd.read_csv(output_dir / 'requests_amr_entity_recog.csv')
# direct = pd.read_csv(output_dir / 'requests_direct_entity_recog.csv')
#
# amr_gold = amr_gold[['text', 'f1']]
# direct = direct[['text', 'f1']]
#
#
#
# # Run t-test on the f1 scores between amr_gold and direct
# amr_gold_f1 = amr_gold['f1']
# direct_f1 = direct['f1']
# t, p = stats.ttest_ind(amr_gold_f1, direct_f1)
# ci = stats.ttest_ind(amr_gold_f1, direct_f1).confidence_interval()
# print('amr_gold vs. direct')
# print('Difference in mean f1 scores between amr_gold and direct')
# print(amr_gold_f1.mean() - direct_f1.mean())
# print(f't = {t}, p = {p}')
#
# print("95 % confidence interval for the difference between amr_gold and direct")
# print(ci)
#
# #
# #
# #
# # # Run t-test on the f1 scores between amr_parser and direct
# #
# amr_parser_f1 = amr_parser['f1']
# t, p = stats.ttest_ind(amr_parser_f1, direct_f1)
# ci = stats.ttest_ind(amr_parser_f1, direct_f1).confidence_interval()
# print('amr_parser vs. direct')
# print('Difference in mean f1 scores between amr_parser and direct')
# print(amr_parser_f1.mean() - direct_f1.mean())
#
# print(f't = {t}, p = {p}')
#
# print("95 % confidence interval for the difference between amr_parser and direct")
# print(ci)
# #
#
# # Run t-test on the f1 scores between amr_gold and amr_parser
# t, p = stats.ttest_ind(amr_gold_f1, amr_parser_f1)
# ci = stats.ttest_ind(amr_gold_f1, amr_parser_f1).confidence_interval()
# print('amr_gold vs. amr_parser')
# print('Difference in mean f1 scores between amr_gold and amr_parser')
# print(amr_gold_f1.mean() - amr_parser_f1.mean())
#
# print(f't = {t}, p = {p}')
#
# print("95 % confidence interval for the difference between amr_gold and amr_parser")
# print(ci)
#
# # Run two-sided t-test on the f1 scores between amr_parser and amr_gold
# t, p = stats.ttest_ind(amr_parser_f1, amr_gold_f1)
# ci = stats.ttest_ind(amr_parser_f1, amr_gold_f1).confidence_interval()
# print('amr_parser vs. amr_gold')
# print('Difference in mean f1 scores between amr_parser and amr_gold')
# print(amr_parser_f1.mean() - amr_gold_f1.mean())
#
# print(f't = {t}, p = {p}')
#
# print("95 % confidence interval for the difference between amr_parser and amr_gold")
# print(ci)

# Run t-test on the f1 scores between logic_amr and logic_direct
direct = pd.read_csv(output_dir / 'gpt-4-0613_remote/requests_direct_logic.csv')
amr = pd.read_csv(output_dir / 'gpt-4-0613_remote/requests_amr_logic.csv')

amr_f1 = amr['score']
direct_f1 = direct['score']

t, p = stats.ttest_ind(amr_f1, direct_f1)
ci = stats.ttest_ind(amr_f1, direct_f1).confidence_interval()
print('logic_amr vs. logic_direct')
print('Difference in mean f1 scores between logic_amr and logic_direct')
print(amr_f1.mean() - direct_f1.mean())

print(f't = {t}, p = {p}')

print("95 % confidence interval for the difference between logic_amr and logic_direct")
print(ci)


# Run t-test on the f1 scores between paws_amr and paws_direct
# paws = pd.read_csv(data_dir / 'featured/paws_features.csv')
paws_amr = pd.read_csv(output_dir / 'gpt-4-0613_remote/requests_amr_paws.csv')
paws_direct = pd.read_csv(output_dir / 'gpt-4-0613_remote/requests_direct_paws.csv')

paws_amr = paws_amr['score']
paws_direct = paws_direct['score']


t, p = stats.ttest_ind(paws_amr, paws_direct)
ci = stats.ttest_ind(paws_amr, paws_direct).confidence_interval()
print('paws_amr vs. paws_direct')
print('Difference in mean f1 scores between paws_amr and paws_direct')
print(paws_amr.mean() - paws_direct.mean())

print(f't = {t}, p = {p}')


# # Run t-test on the f1 scores between logic_amr and logic_direct
# direct = pd.read_csv(output_dir / 'gpt-4-0613_remote/requests_direct_newstest.csv')
# amr = pd.read_csv(output_dir / 'requests_amr_newstest.csv')
#
# amr_f1 = amr['bleu']
# direct_f1 = direct['bleu']
#
# t, p = stats.ttest_ind(amr_f1, direct_f1)
# ci = stats.ttest_ind(amr_f1, direct_f1).confidence_interval()
# print('newtest_amr vs. newtest_direct')
# print('Difference in mean f1 scores between newtest_amr and newstest_direct')
# print(amr_f1.mean() - direct_f1.mean())
#
# print(f't = {t}, p = {p}')

# Run t-test on the f1 scores between logic_amr and logic_direct
direct = pd.read_csv(output_dir / 'gpt-4-0613_remote/requests_direct_pubmed.csv')
amr = pd.read_csv(output_dir / 'gpt-4-0613_remote/requests_amr_pubmed.csv')

amr_f1 = amr['score']
direct_f1 = direct['score']

t, p = stats.ttest_ind(amr_f1, direct_f1)
ci = stats.ttest_ind(amr_f1, direct_f1).confidence_interval()
print('pubmed_amr vs. pubmed_direct')
print('Difference in mean f1 scores between pubmed_amr and pubmed_direct')
print(amr_f1.mean() - direct_f1.mean())

print(f't = {t}, p = {p}')


spider = pd.read_csv(data_dir / 'outputs/spider_files/gpt-4-0613/final_results_all_gpt-4-0613.csv')

spider_amr = spider['exact_match_pred_amr']
spider_direct = spider['exact_match_pred']

#drop the nan values
spider_amr = spider_amr.dropna()
spider_direct = spider_direct.dropna()


t, p = stats.ttest_ind(spider_amr, spider_direct)
ci = stats.ttest_ind(spider_amr, spider_direct).confidence_interval()
print('spider_amr vs. spider_direct')
print('Difference in mean f1 scores between spider_amr and spider_direct')
print(spider_amr.mean() - spider_direct.mean())

print(f't = {t}, p = {p}')

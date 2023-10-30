from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os
import random
# from efficiency.function import set_seed
from scipy.stats import gaussian_kde
import argparse
np.random.seed(0)
random.seed(0)
# set_seed(0)
root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
out_dir = data_dir / "outputs"
ablation_dir = data_dir / "ablation"
parent_dir = os.path.dirname(root_dir)




def summary_stat(df, by_col = 'amr_keep_ratio', save = False):
    mean_values = df.groupby(by_col).mean(numeric_only = True)
    std_values = df.groupby(by_col).std(numeric_only = True)
    # save the summary statistics to a dataframe
    summary_df = mean_values.merge(std_values, left_index = True, right_index = True)
    summary_df = summary_df.rename(columns = {'f1_x':'mean', 'f1_y':'std'})

    return summary_df


def draw_plot(summary_df, save_name = 'cut'):
    # Create KDE plot for 'mean' column
    # sns.kdeplot(data=summary_df, x = 'amr_keep_ratio')
    plt.plot(summary_df.index, summary_df['mean'])
    plt.xlabel('Ratio of AMR Kept')
    plt.ylabel('Average F1 on NER Task')
    # plt.fill_between(summary_df.index, summary_df['mean'] - summary_df['std'], summary_df['mean'] + summary_df['std'],
    #                  color='b', alpha=0.1)
    # Add title and legend
    plt.title('Average F1 on NER Task vs. Ratio of AMR Kept')
    plt.legend()
    plt.savefig(f'{save_name}.pdf', format='pdf')
    plt.savefig(f'{save_name}.png', format='png')

    # Show the plot
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Request to openai models for amr project')

    parser.add_argument('--data_file', type=str, default=data_dir / 'ablation/entity_recog_gold_gpt-4-0613_text.csv', help='the dataset name')
    parser.add_argument('--cut_col', type=str, default='amr', help='which column to cut')
    parser.add_argument('--save', type=bool, default=False, help='Save the output to csv file')
    args = parser.parse_args()
    data_file = args.data_file
    df = pd.read_csv('your_file_path.csv', error_bad_lines=False)

    # df = pd.read_csv(data_file, quotechar='"', delimiter=',', quoting=1, skipinitialspace=True)

    if "text.csv" in str(data_file):
        ratio_col = 'text_keep_ratio'
    else:
        ratio_col = 'amr_keep_ratio'

    summary_stats = summary_stat(df, by_col = ratio_col)
    save_summary = Path(data_file).stem + '_summary.csv'
    if args.save:
        summary_stats.to_csv(save_summary)

    draw_plot(summary_stats, save_name = Path(data_file).stem)


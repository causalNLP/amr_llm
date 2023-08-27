from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os
import random
from efficiency.function import set_seed
from scipy.stats import gaussian_kde

np.random.seed(0)
random.seed(0)
set_seed(0)
root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
out_dir = data_dir / "outputs"
parent_dir = os.path.dirname(root_dir)




def summary_stat(df, by_col = 'amr_keep_ratio', save = False):
    mean_values = df.groupby(by_col).mean(numeric_only = True)
    std_values = df.groupby(by_col).std(numeric_only = True)
    # save the summary statistics to a dataframe
    summary_df = mean_values.merge(std_values, left_index = True, right_index = True)
    summary_df = summary_df.rename(columns = {'f1_x':'mean', 'f1_y':'std'})
    if save:
        summary_df.to_csv(data_dir/'cut_amr_summary.csv')
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
    # plt.savefig(data_dir/f'{save_name}.pdf', format='pdf')
    # plt.savefig(data_dir/f'{save_name}.png', format='png')
    # Show the plot

    # Show the plot
    plt.show()



if __name__ == '__main__':
    # df = pd.read_csv(out_dir/'requests_amr_cutting_entity_recog_true.csv')
    # df_89 = pd.read_csv(out_dir/'requests_amr_cutting_entity_recog_true_8_9.csv')

    # df = df.append(df_89)
    # df.to_csv(out_dir/'requests_amr_cutting_entity_recog_true_large.csv', index=False)
    # df = pd.read_csv(out_dir/'requests_amr_cutting_entity_recog_0719.csv')
    # df = pd.read_csv(out_dir/'requests_amr_cutting_entity_recog_large_0719.csv')
    # df = pd.read_csv(data_dir/'ablations/amr_ablation.csv')
    df = pd.read_csv(data_dir/'ablations/amr_ablation.csv')
    summary_stats = summary_stat(df, by_col = 'amr_keep_ratio', save= True)
    draw_plot(summary_stats, save_name = 'cut_amr_0825')

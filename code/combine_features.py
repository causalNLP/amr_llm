import pandas as pd
import os
from efficiency.function import set_seed
from pathlib import Path
import pdb
import math
import matplotlib.pyplot as plt
import seaborn as sns


set_seed(0)
root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
out_dir = data_dir / "outputs"
parent_dir = os.path.dirname(root_dir)
# Initialize an empty list to store the dataframes


# dfs = []
#
# # Initialize an empty list to store column names of each dataframe
# columns_list = []
#
# # List all files in the './data/featured' directory
# # Note: In a real file system, replace './data/featured' with your actual directory path
# directory_path = data_dir/'featured'
# results_dir = data_dir/'output_gpt4'

# if not os.path.exists(directory_path):
#     os.makedirs(directory_path)
#
#
# # Load all CSV files in the directory
# for filename in os.listdir(directory_path):
#     if filename.endswith('.csv') and filename.split("_")[0] in ['logic','paws','pubmed45','wmt']:
#         filepath = os.path.join(directory_path, filename)
#         df = pd.read_csv(filepath)
#         dfs.append(df)
#
#         # Remove the '_avg' suffix from column names before adding to the list
#         cleaned_columns = {col.replace('_avg', '') for col in df.columns}
#         columns_list.append(cleaned_columns)
#         columns_list.append(set(df.columns))
#
# # Create new dataframes with standardized column names, without modifying the original dataframes
# # This time, we'll check for duplicate names before renaming
# standardized_dfs_checked = []
#
# for df in dfs:
#     new_df = df.copy()
#     new_df = new_df[~new_df['helpfulness'].isna()]

#     new_columns = []
#
#     for col in df.columns:
#         # If renaming would cause a duplicate, keep the original name
#         # If both "ground_truth" and "groundtruth" exist, merge them (Here, we assume they have the same values)
#         if 'ground_truth' in new_df.columns and 'groundtruth' in new_df.columns:
#             new_df['ground_truth'] = new_df['ground_truth'] | new_df['groundtruth']
#             new_df.drop('groundtruth', axis=1, inplace=True)
#         elif 'groundtruth' in new_df.columns:
#             new_df.rename(columns={'groundtruth': 'ground_truth'}, inplace=True)
#
#         new_col = col.replace('_avg', '')
#         if new_col in df.columns and new_col != col:
#             new_columns.append(col)
#         else:
#             new_columns.append(new_col)
#     # If both "ground_truth" and "groundtruth" exist, merge them (Here, we assume they have the same values)
#     if 'ground_truth' in new_df.columns and 'groundtruth' in new_df.columns:
#         new_df['ground_truth'] = new_df['ground_truth'] | new_df['groundtruth']
#         new_df.drop('groundtruth', axis=1, inplace=True)
#     elif 'groundtruth' in new_df.columns:
#         new_df.rename(columns={'groundtruth': 'ground_truth'}, inplace=True)
#     elif 'truth' in new_df.columns:
#         new_df.rename(columns={'truth': 'ground_truth'}, inplace=True)
#     elif 'de' in new_df.columns:
#         new_df.rename(columns={'de': 'ground_truth'}, inplace=True)
#

#     standardized_dfs_checked.append(new_df)
#
# # Concatenate all the standardized dataframes together
# try:
#     concatenated_df_all_columns_checked = pd.concat(standardized_dfs_checked, ignore_index=True)
# except Exception as e:
#     concatenated_df_all_columns_checked = f"An error occurred: {e}"

#
# all = concatenated_df_all_columns_checked
# all.to_csv(data_dir/'all_data_features.csv',index=False)
#
# all = pd.read_csv(data_dir/'all_data_features.csv')
# # Identify columns that have a corresponding '{col_name}_avg' column
# columns_with_avg = [col for col in all.columns if f"{col}_avg" in all.columns]
#
# # Merge the columns
# for col in columns_with_avg:
#     avg_col = f"{col}_avg"
#
#     # Choose non-NaN values from either column
#     all[col] = all[col].combine_first(all[avg_col])
#
#     # Drop the '{col_name}_avg' column
#     all.drop(avg_col, axis=1, inplace=True)
#
#
# all = pd.concat([all,all_spider],ignore_index=True)

# Rename the concatenated dataframe to 'all'
all = pd.read_csv(data_dir/'all_data_features.csv')
#check for duplicated columns

# all = all.loc[:,~all.columns.duplicated()]


# all = concatenated_df_all_columns_checked
# all.to_csv(data_dir/'all_data_features.csv',index=False)

all = pd.read_csv(data_dir/'all_data_features.csv')
# Identify columns that have a corresponding '{col_name}_avg' column
columns_with_avg = [col for col in all.columns if f"{col}_avg" in all.columns]

# Merge the columns
for col in columns_with_avg:
    avg_col = f"{col}_avg"

    # Choose non-NaN values from either column
    all[col] = all[col].combine_first(all[avg_col])

    # Drop the '{col_name}_avg' column
    all.drop(avg_col, axis=1, inplace=True)

# Calculate the percentage of NaN values in each column
nan_percentage = all.isna().mean().round(4) * 100
# Find columns with less than 10% NaN values
columns_less_than_10_percent_nan = nan_percentage[nan_percentage < 10].index.tolist()


# for columns in columns_less_than_10_percent_nan, compute the correlation matrix
# and select top 3 features highly correlated with 'did_llm_fail', 'ground_truth', and 'helpfullness' respectively

# Compute the correlation matrix


# Convert columns to numeric if possible
for col in all.columns:
    all[col] = pd.to_numeric(all[col], errors='ignore')

# Select only numeric columns
numeric_columns = all.select_dtypes(include=['number']).columns.tolist()
# Filter out columns to only keep those with less than 10% NaN values
filtered_df = all[columns_less_than_10_percent_nan]

filtered_df.drop(columns = ['did_llm_failed','ground_truth','amr_improve'],inplace=True)

# Select only numeric columns from the filtered DataFrame
numeric_columns = filtered_df.select_dtypes(include=['number']).columns.tolist()

# Targets to correlate against
targets = ['helpfulness']


# Dictionary to hold the top 5 features with highest absolute value of correlation for each target
top_5_features_dict = {}
top_5_positive_features_dict = {}
top_5_negative_features_dict = {}

# # Compute the Pearson correlation for numerical columns and find the top 5 features for each target
# for target in targets:
#     if target in numeric_columns:
#         # Compute the correlation matrix
#         correlation_matrix = filtered_df[numeric_columns].corr()
#
#         # Sort by the absolute value of the correlation coefficient but keep the original sign
#         # sorted_correlation = correlation_matrix[[target]].apply(lambda x: abs(x)).sort_values(by=target,
#         #                                                                                       ascending=False)
#
#         # Calculate absolute values for sorting
#         abs_sorted_correlation = correlation_matrix[[target]].apply(lambda x: abs(x)).sort_values(by=target,
#                                                                                                   ascending=False)
#
#         # Retain the original signs by reindexing the original correlation matrix
#         sorted_correlation = correlation_matrix[[target]].reindex(abs_sorted_correlation.index)
#
#         # Drop the target itself and other targets from the sorted list
#         sorted_correlation = sorted_correlation.drop(index=[target] + [t for t in targets if t != target])
#
#         # Take the top 5 features
#         top_5_features = sorted_correlation.head(5)
#         # Include both the feature name and the correlation coefficient
#         top_5_features_dict[target] = [(index, row[target]) for index, row in top_5_features.iterrows()]

# Compute the Pearson correlation for numerical columns and find the top 5 features for each target
for target in targets:
    if target in numeric_columns:
        # Compute the correlation matrix
        correlation_matrix = filtered_df[numeric_columns].corr()

        # Calculate absolute values for sorting
        abs_sorted_correlation = correlation_matrix[[target]].apply(lambda x: abs(x)).sort_values(by=target,
                                                                                                  ascending=False)

        # Retain the original signs by reindexing the original correlation matrix
        sorted_correlation = correlation_matrix[[target]].reindex(abs_sorted_correlation.index)

        # Drop the target itself and other targets from the sorted list
        sorted_correlation = sorted_correlation.drop(index=[target] + [t for t in targets if t != target])

        # Split into positive and negative correlations
        positive_correlation = sorted_correlation[sorted_correlation[target] > 0]
        negative_correlation = sorted_correlation[sorted_correlation[target] < 0]

        # Take the top 5 positive features
        top_5_positive_features = positive_correlation.head(5)
        top_5_positive_features_dict[target] = [(index, row[target]) for index, row in top_5_positive_features.iterrows()]

        # Take the top 5 negative features
        top_5_negative_features = negative_correlation.head(5)
        top_5_negative_features_dict[target] = [(index, row[target]) for index, row in top_5_negative_features.iterrows()]

        # Take the top 5 features overall
        top_5_features = sorted_correlation.head(5)
        top_5_features_dict[target] = [(index, row[target]) for index, row in top_5_features.iterrows()]

print("Top 5 features with highest positive correlation with 'helpfulness'")
print(top_5_positive_features_dict)
print("Top 5 features with highest negative correlation with 'helpfulness'")
print(top_5_negative_features_dict)

# print(top_5_features_dict)
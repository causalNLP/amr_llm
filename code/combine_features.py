import pandas as pd
import os
from efficiency.function import set_seed
from pathlib import Path
import pdb
import math


set_seed(0)
root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
out_dir = data_dir / "outputs"
parent_dir = os.path.dirname(root_dir)
# Initialize an empty list to store the dataframes


dfs = []

# Initialize an empty list to store column names of each dataframe
columns_list = []

# List all files in the './data/featured' directory
# Note: In a real file system, replace './data/featured' with your actual directory path
directory_path = data_dir/'featured'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)


# Load all CSV files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv') and filename.split("_")[0] in ['django','logic','paws','pubmed45','wmt']
        filepath = os.path.join(directory_path, filename)
        df = pd.read_csv(filepath)
        dfs.append(df)

        # Remove the '_avg' suffix from column names before adding to the list
        cleaned_columns = {col.replace('_avg', '') for col in df.columns}
        columns_list.append(cleaned_columns)
        columns_list.append(set(df.columns))

# Create new dataframes with standardized column names, without modifying the original dataframes
# This time, we'll check for duplicate names before renaming
standardized_dfs_checked = []

for df in dfs:
    new_df = df.copy()
    new_columns = []

    for col in df.columns:
        # If renaming would cause a duplicate, keep the original name
        # If both "ground_truth" and "groundtruth" exist, merge them (Here, we assume they have the same values)
        if 'ground_truth' in new_df.columns and 'groundtruth' in new_df.columns:
            new_df['ground_truth'] = new_df['ground_truth'] | new_df['groundtruth']
            new_df.drop('groundtruth', axis=1, inplace=True)
        elif 'groundtruth' in new_df.columns:
            new_df.rename(columns={'groundtruth': 'ground_truth'}, inplace=True)

        new_col = col.replace('_avg', '')
        if new_col in df.columns and new_col != col:
            new_columns.append(col)
        else:
            new_columns.append(new_col)
    # If both "ground_truth" and "groundtruth" exist, merge them (Here, we assume they have the same values)
    if 'ground_truth' in new_df.columns and 'groundtruth' in new_df.columns:
        new_df['ground_truth'] = new_df['ground_truth'] | new_df['groundtruth']
        new_df.drop('groundtruth', axis=1, inplace=True)
    elif 'groundtruth' in new_df.columns:
        new_df.rename(columns={'groundtruth': 'ground_truth'}, inplace=True)
    elif 'truth' in new_df.columns:
        new_df.rename(columns={'truth': 'ground_truth'}, inplace=True)
    elif 'de' in new_df.columns:
        new_df.rename(columns={'de': 'ground_truth'}, inplace=True)

    standardized_dfs_checked.append(new_df)

# Concatenate all the standardized dataframes together
try:
    concatenated_df_all_columns_checked = pd.concat(standardized_dfs_checked, ignore_index=True)
except Exception as e:
    concatenated_df_all_columns_checked = f"An error occurred: {e}"

all = concatenated_df_all_columns_checked


# Identify columns that have a corresponding '{col_name}_avg' column
columns_with_avg = [col for col in all.columns if f"{col}_avg" in all.columns]

# Merge the columns
for col in columns_with_avg:
    avg_col = f"{col}_avg"

    # Choose non-NaN values from either column
    all[col] = all[col].combine_first(all[avg_col])

    # Drop the '{col_name}_avg' column
    all.drop(avg_col, axis=1, inplace=True)

# Rename the concatenated dataframe to 'all'
# Calculate the percentage of NaN values in each column
nan_percentage = all.isna().mean().round(4) * 100
# Find columns with less than 10% NaN values
columns_less_than_10_percent_nan = nan_percentage[nan_percentage < 10].index.tolist()



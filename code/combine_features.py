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
    if filename.endswith('.csv'):
        filepath = os.path.join(directory_path, filename)
        df = pd.read_csv(filepath)
        dfs.append(df)
        columns_list.append(set(df.columns))

# Find the columns that are common to all dataframes
common_columns = set.intersection(*columns_list)
print(common_columns)

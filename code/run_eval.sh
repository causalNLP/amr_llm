#!/bin/bash

# Navigate to the directory containing the script, if necessary
# cd /path/to/directory

# Loop through all files matching the specified pattern

for file in ../data/outputs/*/requests_*_newstest*.csv; do
    echo "Processing $file..."
    python eval_gpt.py --data_file "$file" --dataset newstest
done
python eval_gpt.py --data_file ../data/outputs/text-davinci-001/requests_direct_newstest.csv --dataset newstest
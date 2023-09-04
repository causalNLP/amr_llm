#!/bin/bash

# Loop through each .txt file in data/txt_files/
for data_file in data/txt_files/*.txt; do
    # Extract the base name of the file (remove .txt extension)
    base_name=$(basename "$data_file" .txt)

    # Define the output .json file name
#    data_file = "data/txt_files/${base_name}.txt"
    out_file="data/txt_files/${base_name}.json"

    # Run the Python script
    python ~/Desktop/CausalLLMs/perplexity_calculator.py -f "$data_file" --out-file "$out_file"
done

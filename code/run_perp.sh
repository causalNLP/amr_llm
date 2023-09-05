#!/bin/bash
#
## Loop through each .txt file in data/txt_files/
#for data_file in data/txt_files/*.txt; do
#    # Extract the base name of the file (remove .txt extension)
#    base_name=$(basename "$data_file" .txt)
#
#    # Define the output .json file name
##    data_file = "data/txt_files/${base_name}.txt"
#    out_file="data/txt_files/${base_name}.json"
#
#    # Run the Python script
#    python ~/Desktop/CausalLLMs/perplexity_calculator.py -f "$data_file" --out-file "$out_file"
#done


#!/bin/bash

# Export the Python script path as a variable
export SCRIPT_PATH=../CausalLLMs/perplexity_calculator.py

# Define the function to run the script
run_script() {
    data_file=$1
    base_name=$(basename "$data_file" .txt)
    out_file="data/txt_files/${base_name}.json"
    python $SCRIPT_PATH -f "$data_file" --out-file "$out_file"
}

# Export the function so GNU Parallel can use it
export -f run_script

# Run the function in parallel for each .txt file
find data/txt_files/ -name "*.txt" | parallel -j 4 run_script

import argparse
import csv
import os
import pandas as pd
from transition_amr_parser.parse import AMRParser

#os.environ["HF_HOME"] = "/tmp"
#os.environ["TORCH_HOME"] = "/tmp"


def main(input_file, output_file,model):
    parser = AMRParser.from_pretrained(model)
    df=pd.read_csv(input_file)
    for i in range(0,df.shape[0],1):
        data=list(df['text_detok'].values)[i]
        idx=list(df['id'].values)[i]
        tokens, positions = parser.tokenize(data)
        try:
            annotations, machines = parser.parse_sentence(tokens)
            amr = machines.get_amr()
            res=amr.to_penman(jamr=False, isi=True)
            with open(output_file, 'a') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')
                writer.writerow([idx,data,res])
            if i%100==0:
                print(f'Finished {i} sentences', flush = True)
        except Exception as e:
            print(e)
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gets AMR from text')
    parser.add_argument('--input_file', type=str, default="../data/raw_files/ldc_slang_to_amr.csv", help='the input csv file')
    parser.add_argument('--output_file', type=str, default='../processed/AMR3-structbart-L_slang_output.csv',  help='the output csv file')
    parser.add_argument('--model', type=str, default='AMR3-structbart-L', help='the model name')
    args = parser.parse_args()
    main(args.input_file, args.output_file,args.model)

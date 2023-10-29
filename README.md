# LLM + AMR Project

This repo contains the code and data to explore how AMRs can help LLMs. We test the performance of GPT-3.5 on multiple datasets and tasks (paraphrase detection, translation, logical fallacy detection, code generation, NER, Text-to-SQL), and then add AMRs to the prompts and measure the difference in performance. Then we investigate what are the factors that make the use of AMRs helpful to the tasks.


## File Structure

- `code/`: Contains the code to:
    - Get AMRs from sentences
    - Get LLMs' inference performance
    - Train a binary classifier to predict when AMRs help and when LLMs fail
    - Compute the top N most important words for the binary classification using Shapley values
- `data/`: All our data files are in [this google drive folder](https://drive.google.com/drive/folders/1fgjaSuHpt6SfbkolIaT7LUD99BzwdARP?usp=drive_link) (containing the CSVs for all the datasets). The local `data/` folder mainly contains [descriptions to be added].

## Task 0: Get AMRs ###

We use the library [transition-amr-parser](https://github.com/IBM/transition-amr-parser/tree/master) to get AMRs from sentences. The script to get the AMRs can be found in `code/predict_amr.py`. 


## Task 1: Get LLMs' inference performance

This [script](https://github.com/causalNLP/amr/blob/main/code/general_request.py) is used to call the OpenAI API and get the LLMs' inference performance for the selected task.
1. Pass the input data file, the AMR file, the dataset and the amr flag as arguments to the script.

```bash
python code/general_request.py --data_file data/updated_data_input_classifier_input.csv --amr_file data/corrected_amrs.csv --dataset logic --amr_cot
```
2. The prompts are specified in a dictionary inside the file.
3. The outputs are stored in a csv file in "data/outputs/requests_direct_dataset.csv"
4. At the end of the execution we show the test set performance of the LLMs on the given dataset. 

To use [efficiency](https://github.com/zhijing-jin/efficiency/blob/master/README.md) package, which saves gpt queries into a cache automatically, run the following code:
```bash
pip install efficiency
python code/general_request_chatbot.py --data_file data/updated_data_input_classifier_input.csv --amr_file data/corrected_amrs.csv --dataset logic --amr_cot
<!---
### Task 1: Get LLMs' inference performance

1. Substitute the files and change the path at `# TODO: move files to a local path`
2. Change the settings of which model to test at `# TODO: make them as args`
3. Then run the following code

```bash
python code/gpt4.py
```
-->

## Task 2: Binary classification of {When AMRs help, When LLMs fail}

### How to train
This [script](https://github.com/causalNLP/amr/blob/main/code/train_roberta.py) splits the data into train, dev and test sets. It trains a binary classifier to predict when AMRs help or when LLMs fail (depending on the variable you select). It uses the RoBERTa model. It saves the best performing model according to the metric you select and shows the performance on the test and validation sets. The data files are located in [this google drive folder](https://drive.google.com/drive/folders/17pwdiiu7U1oyly8YwMtqCRdu3GBIWT3K). To train the model with the outcome variable "helpfulness", assign the value "helpfulness" to the variable "outcome_variable" in the script. To train it with the outcome variable "did_llm_failed" assign the value "did_llm_failed" to the variable "outcome_variable".

````bash
python train_roberta.py
````
To run only the evaluation part of the previous script run the following code: 
````bash
python evaluate_roberta.py
````
It loads the saved model and shows the performance on the test and validation sets.

## Task 3: Get the most important words for the binary classification using Shapley values
We use the [shap](https://shap.readthedocs.io/en/latest/) library to compute the most influential words for the binary classification. 
 ````bash
python shapley_values.py --model_path translation_model_path --filename data/final_results_trans_corrected.csv --dataset translation --results_path processed/shapley/translation/
````
Given a trained binary classifier, it computes the shapley values for the words in the input sentences. It saves the results by chunks in pkl files. Then it reads them all and compute the top N most important words for the classification. It saves the results in a csv file.

## Task 4: Get a comprehensive list of linguistic features.
We include the features proposed [this](https://github.com/facebookresearch/text_characterization_toolkit) repo, as well as AMR features.
(In current implementation, we assume the text-characterization-toolkit is in the same directory as this repo. ie `../text-characterization-toolkit`)
 ````bash
python get_features  --data_file ../data/to_process/ldc_slang_to_process.csv --dataset ldc_slang --output_file ../data/to_process/ldc_slang_check.csv
````

## Task 5: Get the correlation between linguistic features, LLM performance, and AMR helpfulness .
We include the features proposed [this](https://github.com/facebookresearch/text_characterization_toolkit) repo, as well as AMR features.
 ````bash
python get_corr.py
````


## Task 6: Ablation study: Cutting input text or AMR to see how it affects the performance.
 ````bash
python amr_cot_ablation.py --dataset logic --ratio 0.5 --target amr
````

## Task 7: Effect of ground_truth AMR on LLM performance
An an intermediate step of constructing the GoldAMR-Slang-Para dataset, we let gpt-3.5-turbo-0613 to identify candidate slang usage.
 ````bash
python create_slang.py
````

## Task 8: Human evaluation of LLMs' reasoning ability over AMR
We annotate 50 samples from the PAWS dataset, and ask human annotators to evaluate the correctness of LLMs reasoning over AMR based on the following criteria:
1. The commonalities and differences between the two AMRs are correctly identified.
2. Drawing on the commonalities and differences, the LLMs can correctly infer the relationship between the two sentences.

The annotation results can be found [here](https://docs.google.com/spreadsheets/d/1XXZ88Xwl5O9rWFcyTQxpc3ce7_kf24L-W6oZcphCv_8/edit?usp=sharing).
 ````bash

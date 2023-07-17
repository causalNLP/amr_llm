# LLM + AMR Project

This repo contains the code and data to explore how AMRs can help LLMs. We test the performance of GPT-3.5 on multiple datasets and tasks (paraphrase detection, translation, logical fallacy detection, code generation, NER, Text-to-SQL), and then add AMRs to the prompts and measure the difference in performance. Then we investigate what are the factors that make the use of AMRs helpful to the tasks


## File Structure

- `code/`: Contains the code to:
    - Get AMRs from sentences
    - Get LLMs' inference performance
    - Train a binary classifier to predict when AMRs help and when LLMs fail
- `data/`: All our data files are in [this google drive folder](https://drive.google.com/drive/folders/17pwdiiu7U1oyly8YwMtqCRdu3GBIWT3K) (containing the CSVs for all the datasets). The local `data/` folder mainly contains [descriptions to be added].

## Task 0: Get AMRs ###

We use the library [transition-amr-parser](https://github.com/IBM/transition-amr-parser/tree/master) to get AMRs from sentences. The script to get the AMRs can be found in `code/predict_amr.py`. 


## Task 1: Get LLMs' inference performance

This [script](https://github.com/causalNLP/amr/blob/main/code/general_request.py) is used to call the OPENAI API and get the LLMs' inference performance for the selected task.
1. Pass the input data file, the AMR file, the dataset and the amr flag as arguments to the script.

```bash
python code/general_request.py --data_file data/updated_data_input_classifier_input.csv --amr_file data/corrected_amrs.csv --dataset logic --amr_cot
```
2. The prompts are specified in a dictionary inside the file.
3. The outputs are stored in a csv file in "data/outputs/requests_direct_"+dataset+".csv"
4. At the end of the execution we show the test set performance of the LLMs on the given dataset. 

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
This [script](https://github.com/causalNLP/amr/blob/main/code/train_roberta.py) splits the data into train, dev and test sets. It trains a binary classifier to predict when AMRs help or when LLMs fail (depending on the variable you select). It uses the RoBERTa model. It saves the best performing model according to the metric you select and shows the performance on the test and validation sets. The data files are located in [this google drive folder](https://drive.google.com/drive/folders/17pwdiiu7U1oyly8YwMtqCRdu3GBIWT3K)

````bash
python train_roberta.py
````
Runs only the evaluation part of the previous script. It loads the saved model and shows the performance on the test and validation sets.
````bash
python evaluate_roberta.py
````

## Task 3: Get the most important words for the binary classification using Shapley values
We use the [shap](https://shap.readthedocs.io/en/latest/) library to compute the most influential words for the binary classification. 
 ````bash
python shapley_values.py --model_path translation_model_path --filename data/final_results_trans_corrected.csv --dataset translation --results_path processed/shapley/translation/
````
Given a trained binary classifier, it computes the shapley values for the words in the input sentences. It saves the results by chunks in pkl files. Then it reads them all and compute the top N most important words for the classification. It saves the results in a csv file.


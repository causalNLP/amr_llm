# AMRCoT Project

This repo contains the code and data for the paper: 

[**Analyzing the Role of Semantic Representations in the Era of Large Language Models**](https://zhijing-jin.com/files/papers/AMR_2023.pdf) (2023)
	
*Zhijing Jin\*, Yuen Chen\*, Fernando Gonzalez Adauto\*, Jiayi Zhang, Jiarui Liu, Julian Michael, Bernhard Schölkopf, Mona Diab*       (*: Co-first author)


## File Structure

- `code/`: Contains the codes for the Tasks 0-8 described below.

- `data/`: For the source data, please download the data files from [this google drive folder](https://drive.google.com/drive/folders/1fgjaSuHpt6SfbkolIaT7LUD99BzwdARP?usp=drive_link) (containing the CSVs for all the datasets) to the local `data/` folder. The existing files in the local `data/` folder contains the AMRs of all datasets parsed using [AMR3-structbart-L](https://github.com/IBM/transition-amr-parser), [text input for prompt generation](data/classifier_inputs/updated_data_input%20-%20classifier_input.csv), and [input for Task 2](data/classifier_inputs/data_for_bert.csv) and [default Task 6](data/ldc_ner_features_true.csv).

## Task 0: Get AMRs ###

We use the library [transition-amr-parser](https://github.com/IBM/transition-amr-parser/tree/master) to get AMRs from sentences. The script to get the AMRs can be found in `code/predict_amr.py`. 


## Task 1: Get LLMs' inference performance

To use [efficiency](https://github.com/zhijing-jin/efficiency/blob/master/README.md) package, which saves gpt queries into a cache automatically, run the following code:

```bash
pip install efficiency
````

This [script](code/general_request_chatbot.py) is used to call the OpenAI API and get the LLMs' inference performance for the selected task.

1. Pass the input data file, the AMR file, the dataset, the amr flag, and model version as arguments to the script. For example:

```bash
python code/general_request_chatbot.py --data_file data/classifier_inputs/updated_data_input_classifier_input.csv --amr_file data/corrected_amrs.csv --dataset logic --amr_cot --model_version gpt4
```

To get LLMs' response on SPIDER dataset, run the following code:

```bash
python code/general_request_spider.py --amr_cot --model_version gpt4
````

2. The outputs are stored in a csv file in `data/outputs/{model_version}/requests_direct_{dataset}.csv`
3. To get the results for all the datasets, run the following code:

```bash
python code/eval_gpt.py --data_file {file_to_evaluate} --dataset {dataset}
````

For example:

```bash
python code/eval_gpt.py --data_file data/outputs/gpt-4-0613/requests_direct_logic.csv --dataset logic
```


## Task 2: Binary classification of when AMRs help

### How to train

To train a binary classifier to predict when AMRs help and when LLMs fail,

1. installed the required packages.

```bash
python -r code/BERTBinaryClassification/requirements.txt
````

2. Download this data folder from [google drive](https://drive.google.com/drive/folders/1iYerE6ZXx8fCzvl1LszpUx4YFmJuW1hk?usp=drive_link) and put it under the ```code/BERTBinaryClassification``` directory.

3. Run ```code/BERTBinaryClassification/train.ipynb```.

[//]: #
[//]: # "## Task 3: Get the most important words for the binary classification using Shapley values"

[//]: # "We use the [shap]&#40;https://shap.readthedocs.io/en/latest/&#41; library to compute the most influential words for the binary classification. "

[//]: # " ````bash"

[//]: # "python shapley_values.py --model_path translation_model_path --filename data/final_results_trans_corrected.csv --dataset translation --results_path processed/shapley/translation/"

[//]: # "````"

[//]: # "Given a trained binary classifier, it computes the shapley values for the words in the input sentences. It saves the results by chunks in pkl files. Then it reads them all and compute the top N most important words for the classification. It saves the results in a csv file."

## Task 3: Get a comprehensive list of linguistic features

We generate the features in the Text Characterization Toolkit  (Simig et al., 2022; [this](https://github.com/facebookresearch/text_characterization_toolkit) repo), as well as our own proposed features.

(In current implementation, we assume the text-characterization-toolkit is in the same directory as this repo. ie `../text-characterization-toolkit`)

 ````bash
python code/get_features.py --dataset paws --output_dir ../data/featured
 ````

## Task 4: Get the correlation between linguistic features and AMR helpfulness

We combine all datasets into one csv file, and compute the correlation between linguistic features (features which >90% of the data has) and AMR helpfulness.

````bash
python code/combine_features.py
````

## Task 5: Regress AMR helpfulness on linguistic features (Table 8)

We fit traditional machine learning methods, such as logistic regression, decision tree, random forest, XGBoost, and ensemble models, to predict AMR helpfulness using linguistic features:

 ````bash
 python code/train_basics.py
 ````


## Task 6: Ablation study: Cutting input text or AMR to see how it affects the performance

 ````bash
python amr_cot_ablation.py --dataset entity_recog_gold --cut_col amr --ratio 0.5 --output_dir data/ablation --model_version gpt-4-0613
 ````

The output is stored in a csv file in `{output_dir}/{dataset}_{model_version}_{cutcol}.csv`

To plot the results, run the following code:

 ````bash
python code/plot_ablation.py --data_file ./data/ablation/entity_recog_gold_gpt-4-0613_text.csv --cut_col amr
 ````

The plot is stored in ```data/ablation/{dataset}_{model_version}_{cut_col}.png```
The summary csv is stored in ```data/ablation/{dataset}_{model_version}_{cut_col}_summary.csv```.

## Task 7: Composing the GoldAMR-ComposedSlang dataset

As an intermediate step of constructing the GoldAMR-ComposedSlang dataset, we let gpt-3.5-turbo-0613 to identify candidate slang usage:

 ````bash
python create_slang.py
 ````

## Task 8: Human evaluation of LLMs' reasoning ability over AMR

We annotate 50 samples from the PAWS dataset, and ask human annotators to evaluate the correctness of LLMs reasoning over AMR based on the following criteria:

1. The commonalities and differences between the two AMRs are correctly identified.
2. Drawing on the commonalities and differences, the LLMs can correctly infer the relationship between the two sentences.

The annotation results can be found [here](https://docs.google.com/spreadsheets/d/1XXZ88Xwl5O9rWFcyTQxpc3ce7_kf24L-W6oZcphCv_8/edit?usp=sharing).



### Contact

For **coding and data questions**, 

- Please first open a GitHub issue.
- If you want a more speedy response, please link your GitHub issue when emailing any of the student authors on this paper: [Yuen Chen](https://chenyuen0103.github.io/), [Fernando Gonzalez](https://feradauto.github.io/), and [Jiarui Liu](https://jiarui-liu.github.io/). 
- We will reply to your email and directly answer on the GitHub issue, so more people can benefit if they have similar questions.


For **future collaborations** or further requests,

- Feel free to email [Zhijing Jin](https://zhijing-jin.com) and [Yuen Chen](https://chenyuen0103.github.io/).

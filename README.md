# LLM + AMR Project

This repo contains the code and data to explore how AMRs can help LLMs.



### File Structure

- `code/`: Contains the code to query LLMs (`code/gpt4.py`)
- `data/`: All our data files are in [this google drive folder](https://drive.google.com/drive/folders/17pwdiiu7U1oyly8YwMtqCRdu3GBIWT3K) (containing the CSVs for all the datasets). The local `data/` folder mainly contains [descriptions to be added].



### Task 2: Binary classification of {When AMRs help, When LLMs fail}

#### How to train

````bash
python train_roberta.py
````

This script contains the exact splits for training the binary classifier, and then calls the roberta model.



### Task 1: Get LLMs' inference performance

1. Substitute the files and change the path at `# TODO: move files to a local path`
2. Change the settings of which model to test at `# TODO: make them as args`
3. Then run the following code

```bash
python code/gpt4.py
```

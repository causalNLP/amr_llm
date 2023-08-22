# Evaluation on the Text-to-SQL task (Spider dataset)

The standard way of evaluating this task is using thsi [repo](https://github.com/taoyds/test-suite-sql-eval).
We need to transform our data into the format of the repo and then run the evaluation script. To do that we need to follow the following steps:

1. Follow the instructions in the repo to install the dependencies and download the data.
2. Copy the file `code/spider_evaluation/evaluation.py` to the folder `test-suite-sql-eval/`. We adapted a function to our needs. In specific we store the results of each data point in a CSV file.
3. Run the following script:

```bash
./run_evaluation.sh
```
It first formats our data in the format of the repo and then runs the evaluation script for direct and amr predictions. The results are stored in a CSV file and then merged in a single file.


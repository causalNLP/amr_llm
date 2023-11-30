Get feature importance data and store it to `top_words_dataset.csv`:
-  location: `code/shapley_values.py`
- Cd to this directory, and run this python file: `python3 shapley_values.py --model_path models/20231015-07:04:23_binary_0.1_16_1e-05_adamw.pt --filename data/test_label1.csv --dataset dataset --results_path data/new_als/`

Postprocessing feature importance data:
- `python3 code/test.py`
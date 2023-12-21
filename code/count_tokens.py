import tiktoken
import pandas as pd
from scipy import stats
from pathlib import Path
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4")
root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
# output_dir = data_dir / "output_gpt4"
output_dir = data_dir / "outputs"
paws_fewshot_amr = pd.read_csv(output_dir / 'gpt-3.5-turbo-0613/requests_amr_paws_few_shot.csv')
paws_fewshot_direct = pd.read_csv(output_dir / 'gpt-3.5-turbo-0613/requests_direct_paws_few_shot.csv')

# calculate the averge number of tokens row['raw_prompt'] for each row in paws_fewshot_amr

paws_fewshot_amr['num_tokens'] = paws_fewshot_amr['raw_prompt'].apply(lambda x: len(enc.encode(x)))
paws_fewshot_direct['num_tokens'] = paws_fewshot_direct['raw_prompt'].apply(lambda x: len(enc.encode(x)))

#print the average number of tokens for each dataset
print("Avg # tokens in paws amr 2-shot",paws_fewshot_amr['num_tokens'].mean())
print("Avg # tokens in paws direct 2-shot", paws_fewshot_direct['num_tokens'].mean())


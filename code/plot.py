from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
import random
np.random.seed(0)
random.seed(0)

root_dir = Path(__file__).parent.parent.resolve()
current_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"
out_dir = data_dir / "outputs"
tct_out_dir = data_dir / "tct_outputs"
parent_dir = os.path.dirname(root_dir)
good_dir = data_dir / "good"
prediction_dir = data_dir / "predictions"
onto_dir = f'{parent_dir}/ontonotes-release-5.0'
model_dir = root_dir / "model"
sample_dir = data_dir / "samples"
google_dir = r"~/Google Drive/My Drive/Zhijing&Yuen/"
google_amr_data_dir = r"~/Google Drive/My Drive/Zhijing&Yuen/amr_codes/data/"
google_pred_dir = r"~/Google Drive/My Drive/Zhijing&Yuen/amr_codes/data/predictions"


df = pd.read_csv(out_dir/'requests_amr_cutting_entity_recog_true.csv')
df_89 = pd.read_csv(out_dir/'requests_amr_cutting_entity_recog_true_8_9.csv')

df = df.append(df_89)
df.to_csv(out_dir/'requests_amr_cutting_entity_recog_true.csv', index=False)
mean_values = df.groupby('amr_keep_ratio').mean()

# Create a new figure
plt.figure()

# Plot the mean values
plt.plot(mean_values.index, mean_values.values)

# Add labels
plt.title('Average F1 on NER Task vs. Ratio of AMR Kept')
plt.xlabel('Ratio of AMR Kept')
plt.ylabel('Average F1')

# Save the figure as a PDF
plt.savefig('cut.pdf', format='pdf')

# Show the plot
plt.show()
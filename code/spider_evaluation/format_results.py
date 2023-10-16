import pandas as pd
import numpy as np

res=pd.read_csv("./indiv_results.csv")
res_amr=pd.read_csv("./indiv_results_amr.csv")

dz=pd.read_csv("../../../processed/files/final_results_spider_corrected.csv")

#dz=dz.loc[dz.id.str.contains('dev')]

res=res.loc[:,['pred_formatted_eval','exact_match_pred']]
res_amr=res_amr.loc[:,['pred_amr_formatted_eval','exact_match_pred_amr']]

df_final=pd.concat([dz,res,res_amr],axis=1)

print("Exact match direct:",df_final.exact_match_pred.mean())
print("Exact match amr:",df_final.exact_match_pred_amr.mean())

df_final.loc[:,['id', 'text_detok', 'amr', 'ground_truth', 'pred', 'pred_amr', 'exact_match_pred', 'exact_match_pred_amr']]

df_final.to_csv("../../data/outputs/final_results_spider_corrected_evaluated.csv",index=False)
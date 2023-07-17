import pandas as pd
import numpy as np
import subprocess

dz=pd.read_csv("../../../processed/files/final_results_spider_corrected.csv")
dev_spider=pd.read_json("../../data/dev_spider.json")

dev_spider=dev_spider.loc[:,['db_id','question']].rename(columns={'question':'text_detok'})

dz=dz.merge(dev_spider,on=['text_detok'])

dz=dz.loc[dz.id.str.contains('dev')]


gold=dz.loc[:,['ground_truth','db_id']]
pred_amr=dz.loc[:,['pred_amr']]
pred=dz.loc[:,['pred']]

pred.pred=pred.pred.replace('\n',' ',regex=True)
pred_amr.pred_amr=pred_amr.pred_amr.replace('\n',' ',regex=True)
gold.ground_truth=gold.ground_truth.replace('"',"'",regex=True)

pred=pred.reset_index(drop=True)
pred_amr=pred_amr.reset_index(drop=True)
gold=gold.reset_index(drop=True)


gold.to_csv("../../data/outputs/gold_spider.txt",sep='\t',index=False,header=False)

pred.to_csv("../../data/outputs/pred_spider_direct.txt",sep='\t',index=False,header=False)

pred_amr.to_csv("../../data/outputs/pred_spider_amr.txt",sep='\t',index=False,header=False)
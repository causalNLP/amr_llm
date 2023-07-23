import pandas as pd
import numpy as np
import subprocess

dz=pd.read_csv("../../../processed/files/final_results_spider_corrected.csv")
dev_spider=pd.read_json("../../data/dev_spider.json")
train_spider=pd.read_json("../../data/train_spider.json")

dev_spider=pd.concat([dev_spider,train_spider])
dev_spider['query']=dev_spider['query'].replace("\t","",regex=True)

dz['ground_truth']=dz['ground_truth'].replace("\t","",regex=True)

dev_spider=dev_spider.loc[:,['db_id','question']].rename(columns={'question':'text_detok_db'})
dev_spider=dev_spider.reset_index(drop=True)
dz=dz.reset_index(drop=True)

df_final=pd.concat([dz,dev_spider],axis=1)

assert df_final.loc[df_final.text_detok!=df_final.text_detok_db].shape[0]==0
#dz=dz.loc[dz.id.str.contains('dev')]


gold=df_final.loc[:,['ground_truth','db_id']]
pred_amr=df_final.loc[:,['pred_amr']]
pred=df_final.loc[:,['pred']]

pred.pred=pred.pred.replace('\n',' ',regex=True)
pred_amr.pred_amr=pred_amr.pred_amr.replace('\n',' ',regex=True)
gold.ground_truth=gold.ground_truth.replace('"',"'",regex=True)

pred=pred.reset_index(drop=True)
pred_amr=pred_amr.reset_index(drop=True)
gold=gold.reset_index(drop=True)


gold.to_csv("../../data/outputs/gold_spider.txt",sep='\t',index=False,header=False)

pred.to_csv("../../data/outputs/pred_spider_direct.txt",sep='\t',index=False,header=False)

pred_amr.to_csv("../../data/outputs/pred_spider_amr.txt",sep='\t',index=False,header=False)
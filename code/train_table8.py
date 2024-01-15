from sklearn.model_selection import train_test_split
import pandas as pd

class Train():
    def __init__(self, df, model_type):
        self.df = df
        self.model_type = model_type
    
    
    def resplit_df(self):
        # 4:1:1 -> 
        category_list = ['pubmed45', 'paws', 'spider', 'newstest16', 'newstest13', 'logic']
        category_df = []
        
        
        for category in category_list:
            tmp_df = self.df.loc[self.df['id'].str.split("_", expand=True)[0] == category]
            tmp_train, tmp_val_test = train_test_split(tmp_df, test_size=0.33, random_state=42)
            tmp_val, tmp_test = train_test_split(tmp_val_test, test_size=0.5, random_state=42)
            category_df.append((tmp_train, tmp_val, tmp_test))
        
        df_train, df_val, df_test = pd.concat([i[0] for i in category_df], ignore_index=True, axis=0), pd.concat([i[1] for i in category_df], ignore_index=True, axis=0), pd.concat([i[2] for i in category_df], ignore_index=True, axis=0)
        df_train = df_train.sample(frac=1, random_state=42)
        df_train.reset_index(drop=True, inplace=True)
        df_val = df_val.sample(frac=1, random_state=42)
        df_val.reset_index(drop=True, inplace=True)
        df_test = df_test.sample(frac=1, random_state=42)
        df_test.reset_index(drop=True, inplace=True)
        df_train['split'], df_val['split'], df_test['split'] = 'train', 'val', 'test'
        print("Train size: ", len(df_train))
        print("Val size: ", len(df_val))
        print("Test size: ", len(df_test))
        return df_train, df_val, df_test

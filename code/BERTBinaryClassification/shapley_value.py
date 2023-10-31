from sklearn.feature_extraction import DictVectorizer
import numpy as np
from slicer import Slicer
import pickle


def convert_shap_values(shap_values, sorted_feature_names_list=None):

    # assume that data is a list of dictionaries
    data = []  # a list of dict
    for shap_tuple in shap_values:
        tmp_dict = {}
        for idx, val in enumerate(shap_tuple.values.tolist()):
            tmp_dict[shap_tuple.data[idx].strip()] = val
        data.append(tmp_dict)

    # create a DictVectorizer object and fit it to the data
    vec = DictVectorizer(sparse=True)
    vec.fit(data)

    # transform the data into a sparse matrix with counts
    X = vec.transform(data).toarray()
    feature_names = vec.get_feature_names()

    if sorted_feature_names_list is not None:
        index_list = [feature_names.index(x)
                      for x in sorted_feature_names_list]
        feature_names = sorted_feature_names_list
        X = np.take(X, index_list, axis=1)

    return X, feature_names


def build_new_shap_values(X_arr, feature_names, explanation):
    # explanation: shap._explanation.Explanation
    # X_counts is an array with the count of each feature across all examples
    shap_values = explanation(
        X_arr,
        data=np.tile(feature_names, X_arr.shape[0]).reshape(
            X_arr.shape[0], -1),
        feature_names=feature_names
    )
    return shap_values


def build_spreadsheet(withheld_list, top=None):
    shap_list = []
    withheld_real_list = []
    for withheld in withheld_list:
        with open(f"/nfs/turbo/coe-vvh/ljr/Censorship/shap_data/beeswarm/v0328_shap_vals_test_uniq_en_{withheld}_samp500.pkl", 'rb') as f:
            shap_list.append(pickle.load(f))
        # build real name list
        if withheld == 'None':
            withheld_real_list.append("all")
        else:
            withheld_real_list.append(withheld)

    import shap_customized.shap as shap
    import pandas as pd

    old_df = pd.DataFrame()
    for withheld_idx, (withheld, shap_values) in enumerate(zip(withheld_real_list, shap_list)):
        df_list = []
        tmp_all = shap_values._numpy_func(
            "mean", **{"axis": 0, "mean_over_all": True})
        tmp_n_all = shap_values._numpy_func(
            "mean", **{"axis": 0, "mean_over_all": False})
        # print(tmp_all[0])
        # print(tmp_n_all[0])
        assert np.sum(~(tmp_all[0] == tmp_n_all[0])) == 0

        if top is None:
            for idx, feature_name in enumerate(tmp_all[0]):
                df_list.append({"feature_name": feature_name,
                                f"{withheld}_samp_mean_over_all": tmp_all[1][idx], f"{withheld}_samp_mean_over_nonzero": tmp_n_all[1][idx]})
        # if top:
        else:
            # sort by abs from max to min
            tmp_all_idx = np.argsort(np.abs(tmp_all[1]))[::-1][:top]
            tmp_all_0 = tmp_all[0][tmp_all_idx]
            tmp_all_1 = tmp_all[1][tmp_all_idx]

            tmp_n_all_idx = np.argsort(np.abs(tmp_n_all[1]))[::-1][:top]
            tmp_n_all_0 = tmp_n_all[0][tmp_n_all_idx]
            tmp_n_all_1 = tmp_n_all[1][tmp_n_all_idx]

            df_tmp_list = []
            for idx, feature_name in enumerate(tmp_all_0):
                df_tmp_list.append({"feature_name": feature_name,
                                    f"{withheld}_samp_mean_over_all": tmp_all_1[idx]})
            df_tmp_all = pd.DataFrame().from_records(df_tmp_list)

            df_tmp_list = []
            for idx, feature_name in enumerate(tmp_n_all_0):
                df_tmp_list.append({"feature_name": feature_name,
                                    f"{withheld}_samp_mean_over_nonzero": tmp_n_all_1[idx]})
            df_tmp_n_all = pd.DataFrame().from_records(df_tmp_list)

            df_tmp_merge = pd.merge(
                df_tmp_all, df_tmp_n_all, on='feature_name', how='outer')
            df_list = df_tmp_merge.to_dict(orient='records')

        df = pd.DataFrame().from_records(df_list)
        if withheld_idx >= 1:
            new_df = df
            old_df = pd.merge(old_df, new_df, on='feature_name', how='outer')
        else:
            old_df = df

    return old_df


def get_text_samples(df, feature_list):
    def get_url_by_id(id):
        if id == 0:
            return None
        url = 'https://twitter.com/Unknown/status/' + str(id)
        return url
    text_sample_0, text_sample_1, text_sample_2 = [], [], []
    # cnt = 0
    for feature in feature_list:

        # print(feature)
        if type(feature) == float:
            # print("true")
            text_sample_0.append(None)
            text_sample_1.append(None)
            text_sample_2.append(None)
        else:
            # print(feature)
            # tmp = df.loc[df['full_text_wo_label'].fillna("").str.contains(
            #     feature), 'full_text_wo_label'].values.tolist()
            tmp = df.loc[df['full_text_wo_label'].fillna("").str.contains(
                feature), 'id'].fillna(0).astype(int).values.tolist()
            if len(tmp) >= 3:
                text_sample_0.append(get_url_by_id(tmp[0]))
                text_sample_1.append(get_url_by_id(tmp[1]))
                text_sample_2.append(get_url_by_id(tmp[2]))
            elif len(tmp) == 2:
                text_sample_0.append(get_url_by_id(tmp[0]))
                text_sample_1.append(get_url_by_id(tmp[1]))
                text_sample_2.append(None)
            elif len(tmp) == 1:
                text_sample_0.append(get_url_by_id(tmp[0]))
                text_sample_1.append(None)
                text_sample_2.append(None)
            else:
                text_sample_0.append(None)
                text_sample_1.append(None)
                text_sample_2.append(None)
        # print(tmp)
        # cnt += 1
        # if cnt == 3:
        #     break
    return text_sample_0, text_sample_1, text_sample_2

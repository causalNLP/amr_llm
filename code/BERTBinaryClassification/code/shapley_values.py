import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import pandas as pd
import pickle
import shap
import torch.nn.functional as F
import transformers
from sklearn.model_selection import train_test_split
import os
import csv
import matplotlib.pyplot as pl
import scipy
import argparse
from shap import Cohorts,Explanation
from shap.utils import format_value, ordinal_str
from shap.utils._exceptions import DimensionError
from shap.plots import colors
from shap.plots._labels import labels
from shap.plots._utils import (
    convert_ordering,
    dendrogram_coords,
    get_sort_order,
    merge_nodes,
    sort_inds,
)
import re


def bar_customized(shap_values, max_display=10, order=Explanation.abs, clustering=None, clustering_cutoff=0.5,
        merge_cohorts=False, show_data="auto", show=True,top_number=10):
    """
    Customized barplot that return the top features with the highest absolute shap values
    """
    if isinstance(shap_values, Explanation):
        cohorts = {"": shap_values}
    elif isinstance(shap_values, Cohorts):
        cohorts = shap_values.cohorts
    elif isinstance(shap_values, dict):
        cohorts = shap_values
    else:
        emsg = (
            "The shap_values argument must be an Explanation object, Cohorts "
            "object, or dictionary of Explanation objects!"
        )
        raise TypeError(emsg)

    # unpack our list of Explanation objects we need to plot
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i, exp in enumerate(cohort_exps):
        if not isinstance(exp, Explanation):
            emsg = (
                "The shap_values argument must be an Explanation object, Cohorts "
                "object, or dictionary of Explanation objects!"
            )
            raise TypeError(emsg)

        if len(exp.shape) == 2:
            # collapse the Explanation arrays to be of shape (#features,)
            cohort_exps[i] = exp.abs.mean(0)
        if cohort_exps[i].shape != cohort_exps[0].shape:
            emsg = (
                "When passing several Explanation objects, they must all have "
                "the same number of feature columns!"
            )
            raise DimensionError(emsg)
        # TODO: check other attributes for equality? like feature names perhaps? probably clustering as well.

    # unpack the Explanation object
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    if clustering is None:
        partition_tree = getattr(cohort_exps[0], "clustering", None)
    elif clustering is False:
        partition_tree = None
    else:
        partition_tree = clustering
    if partition_tree is not None:
        assert partition_tree.shape[1] == 4, "The clustering provided by the Explanation object does not seem to be a partition tree (which is all shap.plots.bar supports)!"
    op_history = cohort_exps[0].op_history
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])

    if len(values[0]) == 0:
        raise Exception("The passed Explanation is empty! (so there is nothing to plot)")

    # we show the data on auto only when there are no transforms
    if show_data == "auto":
        show_data = len(op_history) == 0

    # TODO: Rather than just show the "1st token", "2nd token", etc. it would be better to show the "Instance 0's 1st but", etc
    if issubclass(type(feature_names), str):
        feature_names = [ordinal_str(i)+" "+feature_names for i in range(len(values[0]))]

    # build our auto xlabel based on the transform history of the Explanation object
    xlabel = "SHAP value"
    for op in op_history:
        if op["name"] == "abs":
            xlabel = "|"+xlabel+"|"
        elif op["name"] == "__getitem__":
            pass # no need for slicing to effect our label, it will be used later to find the sizes of cohorts
        else:
            xlabel = str(op["name"])+"("+xlabel+")"

    # find how many instances are in each cohort (if they were created from an Explanation object)
    cohort_sizes = []
    for exp in cohort_exps:
        for op in exp.op_history:
            if op.get("collapsed_instances", False): # see if this if the first op to collapse the instances
                cohort_sizes.append(op["prev_shape"][0])
                break


    # unwrap any pandas series
    if str(type(features)) == "<class 'pandas.core.series.Series'>":
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # ensure we at least have default feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values[0]))])

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(values[0]))
    max_display = min(max_display, num_features)

    # iteratively merge nodes until we can cut off the smallest feature values to stay within
    # num_features without breaking a cluster tree
    orig_inds = [[i] for i in range(len(values[0]))]
    orig_values = values.copy()
    while True:
        feature_order = np.argsort(np.mean([np.argsort(convert_ordering(order, Explanation(values[i]))) for i in range(values.shape[0])], 0))
        if partition_tree is not None:

            # compute the leaf order if we were to show (and so have the ordering respect) the whole partition tree
            clust_order = sort_inds(partition_tree, np.abs(values).mean(0))

            # now relax the requirement to match the parition tree ordering for connections above clustering_cutoff
            dist = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(partition_tree))
            feature_order = get_sort_order(dist, clust_order, clustering_cutoff, feature_order)

            # if the last feature we can display is connected in a tree the next feature then we can't just cut
            # off the feature ordering, so we need to merge some tree nodes and then try again.
            if max_display < len(feature_order) and dist[feature_order[max_display-1],feature_order[max_display-2]] <= clustering_cutoff:
                #values, partition_tree, orig_inds = merge_nodes(values, partition_tree, orig_inds)
                partition_tree, ind1, ind2 = merge_nodes(np.abs(values).mean(0), partition_tree)
                for i in range(len(values)):
                    values[:,ind1] += values[:,ind2]
                    values = np.delete(values, ind2, 1)
                    orig_inds[ind1] += orig_inds[ind2]
                    del orig_inds[ind2]
            else:
                break
        else:
            break

    # here we build our feature names, accounting for the fact that some features might be merged together
    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds), 0, -1)
    feature_names_new = []
    for pos,inds in enumerate(orig_inds):
        if len(inds) == 1:
            feature_names_new.append(feature_names[inds[0]])
        else:
            full_print = " + ".join([feature_names[i] for i in inds])
            if len(full_print) <= 40:
                feature_names_new.append(full_print)
            else:
                max_ind = np.argmax(np.abs(orig_values).mean(0)[inds])
                feature_names_new.append(feature_names[inds[max_ind]] + " + %d other features" % (len(inds)-1))
    feature_names = feature_names_new

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features < len(values[0]):
        num_cut = np.sum([len(orig_inds[feature_order[i]]) for i in range(num_features-1, len(values[0]))])
        values[:,feature_order[num_features-1]] = np.sum([values[:,feature_order[i]] for i in range(num_features-1, len(values[0]))], 0)

    # build our y-tick labels
    yticklabels = []
    for i in feature_inds:
        if features is not None and show_data:
            yticklabels.append(format_value(features[i], "%0.03f") + " = " + feature_names[i])
        else:
            yticklabels.append(re.sub(r"[( ,]", "", feature_names[i]))
    if num_features < len(values[0]):
        yticklabels[-1] = "Sum of %d other features" % num_cut

    # compute our figure size based on how many features we are showing
    row_height = 0.5
    pl.gcf().set_size_inches(8, num_features * row_height * np.sqrt(len(values)) + 1.5)

    # if negative values are present then we draw a vertical line to mark 0, otherwise the axis does this for us...
    negative_values_present = np.sum(values[:,feature_order[:num_features]] < 0) > 0
    if negative_values_present:
        pl.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)

    # draw the bars
    patterns = (None, '\\\\', '++', 'xx', '////', '*', 'o', 'O', '.', '-')
    total_width = 0.7
    bar_width = total_width / len(values)
    for i in range(len(values)):
        ypos_offset = - ((i - len(values) / 2) * bar_width + bar_width / 2)
        pl.barh(
            y_pos + ypos_offset, values[i,feature_inds],
            bar_width, align='center',
            color=[colors.blue_rgb if values[i,feature_inds[j]] <= 0 else colors.red_rgb for j in range(len(y_pos))],
            hatch=patterns[i], edgecolor=(1,1,1,0.8), label=f"{cohort_labels[i]} [{cohort_sizes[i] if i < len(cohort_sizes) else None}]"
        )

    # draw the yticks (the 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks)
    pl.yticks(list(y_pos) + list(y_pos + 1e-8), yticklabels + [l.split('=')[-1] for l in yticklabels], fontsize=13)

    xlen = pl.xlim()[1] - pl.xlim()[0]
    fig = pl.gcf()
    ax = pl.gca()
    #xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen/width

    for i in range(len(values)):
        ypos_offset = - ((i - len(values) / 2) * bar_width + bar_width / 2)
        for j in range(len(y_pos)):
            ind = feature_order[j]
            if values[i,ind] < 0:
                pl.text(
                    values[i,ind] - (5/72)*bbox_to_xscale, y_pos[j] + ypos_offset, format_value(values[i,ind], '%+0.02f'),
                    horizontalalignment='right', verticalalignment='center', color=colors.blue_rgb,
                    fontsize=12
                )
            else:
                pl.text(
                    values[i,ind] + (5/72)*bbox_to_xscale, y_pos[j] + ypos_offset, format_value(values[i,ind], '%+0.02f'),
                    horizontalalignment='left', verticalalignment='center', color=colors.red_rgb,
                    fontsize=12
                )

    # put horizontal lines for each feature row
    for i in range(num_features):
        pl.axhline(i+1, color="#888888", lw=0.5, dashes=(1, 5), zorder=-1)

    if features is not None:
        features = list(features)

        # try and round off any trailing zeros after the decimal point in the feature values
        for i in range(len(features)):
            try:
                if round(features[i]) == features[i]:
                    features[i] = int(features[i])
            except Exception:
                pass # features[i] must not be a number

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    if negative_values_present:
        pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params('x', labelsize=11)

    xmin,xmax = pl.gca().get_xlim()
    ymin,ymax = pl.gca().get_ylim()

    if negative_values_present:
        pl.gca().set_xlim(xmin - (xmax-xmin)*0.05, xmax + (xmax-xmin)*0.05)
    else:
        pl.gca().set_xlim(xmin, xmax + (xmax-xmin)*0.05)

    # if features is None:
    #     pl.xlabel(labels["GLOBAL_VALUE"], fontsize=13)
    # else:
    pl.xlabel(xlabel, fontsize=13)

    if len(values) > 1:
        pl.legend(fontsize=12)

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = pl.gca().yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")

    # draw a dendrogram if we are given a partition tree
    if partition_tree is not None:

        # compute the dendrogram line positions based on our current feature order
        feature_pos = np.argsort(feature_order)
        ylines,xlines = dendrogram_coords(feature_pos, partition_tree)

        # plot the distance cut line above which we don't show tree edges
        xmin,xmax = pl.xlim()
        xlines_min,xlines_max = np.min(xlines),np.max(xlines)
        ct_line_pos = (clustering_cutoff / (xlines_max - xlines_min)) * 0.1 * (xmax - xmin) + xmax
        pl.text(
            ct_line_pos + 0.005 * (xmax - xmin), (ymax - ymin)/2, "Clustering cutoff = " + format_value(clustering_cutoff, '%0.02f'),
            horizontalalignment='left', verticalalignment='center', color="#999999",
            fontsize=12, rotation=-90
        )
        l = pl.axvline(ct_line_pos, color="#dddddd", dashes=(1, 1))
        l.set_clip_on(False)

        for (xline, yline) in zip(xlines, ylines):

            # normalize the x values to fall between 0 and 1
            xv = (np.array(xline) / (xlines_max - xlines_min))

            # only draw if we are not going past distance threshold
            if np.array(xline).max() <= clustering_cutoff:

                # only draw if we are not going past the bottom of the plot
                if yline.max() < max_display:
                    l = pl.plot(
                        xv * 0.1 * (xmax - xmin) + xmax,
                        max_display - np.array(yline),
                        color="#999999"
                    )
                    for v in l:
                        v.set_clip_on(False)

    if show:
        pl.show()
        pl.savefig("my_plot.pdf", format='pdf', bbox_inches="tight")
    top_features=[feature_names[i] for i in list(feature_order[:top_number])]
    top_values=[values[0][i] for i in list(feature_order[:top_number])]
    top_words=list(zip(top_features, top_values))
    return top_words

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def list_files(directory):
    files=[]
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files


def split_sets(dataset,df):
    """Split data into train, dev and test sets, formatting depends on the dataset"""
    if dataset in ['translation']:
        df['set']=df.id.str[:10]
        train_set=df.loc[df['set']=='newstest13']
        dev_set, test_set = train_test_split(df.loc[df['set']=='newstest16'], test_size=0.5,random_state=42)
    elif dataset in ['PAWS','pubmed']:
        train_set, val_df = train_test_split(df, test_size=0.3,random_state=42)
        dev_set, test_set = train_test_split(val_df, test_size=0.5,random_state=42)
    elif dataset in ['logic','django','spider']:
        train_set=df.loc[df['id'].str.contains('train')]
        test_set=df.loc[df['id'].str.contains('test')]
        dev_set=df.loc[df['id'].str.contains('dev')]
    
    return train_set,dev_set,test_set

def compute_shap(test_set,results_path_total,explainer,step=100):
    for i in range(0,test_set.shape[0],step):
        shap_values = explainer(test_set.loc[i:i+step-1].input.values)
        with open(results_path_total+str(i)+'.pkl', 'wb') as file:
            pickle.dump(shap_values, file)

def read_shap_values(results_path):
    files=list_files(results_path)
    with open(results_path+files[0], 'rb') as file:
        contents = CPU_Unpickler(file).load()
    for fi in files[1:]:
        with open(results_path+fi, 'rb') as file:
            contents1 = CPU_Unpickler(file).load()
        contents.values=np.concatenate((contents.values, contents1.values))
        contents.data=contents.data+contents1.data
        contents.feature_names=contents.feature_names+contents1.feature_names
    return contents

def main(model_path, filename, dataset, results_path):
    torch.set_grad_enabled(False)
    results_path_total=results_path+dataset+'_direct_test_'
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base').to("cuda:0")

    model.load_state_dict(torch.load("models/20231015-06:20:43_binary_0.1_16_1e-05_adamw.pt"))
    # df = pd.read_csv(filename)
    # train_set,dev_set,test_set=split_sets(dataset,df)
    # test_set=test_set.loc[:,['id','text']].reset_index(drop=True)
    test_set = pd.read_csv(filename)
    test_set = test_set.loc[:,['id','input']].reset_index(drop=True)

    def f(x):
        tv = torch.tensor([tokenizer.encode(v, pad_to_max_length=True, max_length=512) for v in x]).to("cuda:0")
        outputs = model(tv)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        # val = sp.special.logit(scores[:,1]) # use one vs rest logit units
        # modify val
        val = scores[:, 1]
        return val

    # explainer = shap.Explainer(f, shap.maskers.Text())
    # compute_shap(test_set,results_path_total,explainer,step=100)

    contents=read_shap_values(results_path)
    print(contents.__dict__.keys())
    top_words=bar_customized(contents.sum(0), max_display=11,show=True,top_number=100)

    with open('top_words_'+dataset+'.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['word','value'])
        csv_out.writerows(top_words)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compute shap values')
    parser.add_argument('--model_path', type=str, default='../../processed/modeling/models/translation_hyp_final_helpfulness', help='the path to the model')
    parser.add_argument('--filename', type=str, default='../../processed//files/final_results_trans_corrected.csv', help='the csv file')
    parser.add_argument('--dataset', type=str, default='translation', help='the dataset name')
    parser.add_argument('--results_path', type=str, default='../../processed/shapley/translation/', help='the path to the results')
    args = parser.parse_args()
    #model_path='./models/roberta_trans_weights'
    #filename='./final_results_trans_corrected.csv'
    #dataset='translation'
    #results_path='./shap_res/'+dataset
    
    main(args.model_path, args.filename, args.dataset, args.results_path)
    

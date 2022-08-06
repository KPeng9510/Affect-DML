import umap
from sklearn.decomposition import TruncatedSVD

import sys
from sklearn.manifold import TSNE
import pickle
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import fire
from scipy.cluster.vq import vq, kmeans, whiten
import pandas as pd
#import seaborn as sns 

np.set_printoptions(threshold=sys.maxsize)

#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

ntu_one_shot_classes = [
    'Disconnection',
    'Engagement',
    'Sensitive',
    'Embarrassment',
    'Esteem',
    'Yearning']

ntu_one_shot_classes_ids = range(7,13)
#ntu_one_shot_classes = range(1,21)

def plot_embedding_1(filename='/home/kpeng/oneshot_metriclearning/emotic_dml/models/resnet50_emolevel_66_all_re512_stepsize_10_NTU_ONE_SHOT_SIGNALS_model_resnet18_cl_cross_entropy_ml_triplet_margin_miner_multi_similarity_mix_ml_0.50_mix_cl_0.50_resize_256_emb_size_128_class_size_5_opt_rmsprop_lr_0.00_m_0.00_wd_0.00.pkl_last', target_filename='tsne.png', mode="tsne", legend=True, size=0.03, color_map="utdmhad"):
    """
    Plots an embedding from a dumped pickel file

    :filename: filename to the pickle file
    :target_filename: output file, different extention fives different format
    :mode: could be either `umap` or `tsne`, depending on your preference
    :legend: if 'True' plots a legend
    """
    print(len(ntu_one_shot_classes))
    with open(filename, "rb") as f:
        a = pickle.load(f)
    
    num_samples = len(a[0])

    if mode == "tsne":

        X_Train_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(a[0])
        tsne =  TSNE(n_components=2,  perplexity=35, early_exaggeration=14, learning_rate=0.06, n_iter=1000)
        x_reduced=tsne.fit_transform(X_Train_reduced)
        #x_reduced=pd.DataFrame(tsne.embedding_,index=X_Train_reduced.index)
        
        #print(x_reduced)
    if mode == "umap":
        x_reduced = umap.UMAP(random_state=42).fit_transform(a[0])
    y = a[1][:num_samples].flatten()
    #print(y)
    print(x_reduced.shape, y.shape)
    

    #plt.figure(figsize=(, 5))
    plt.rcParams['legend.fontsize'] = 'xx-small'
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    if color_map == "ntu":
        colors = plt.cm.get_cmap("tab20b").colors
    else:
        colors = plt.cm.get_cmap("Set1").colors
    #colors = plt.cm.get_cmap("hsv", 20).colors
    for i, c, label in zip(ntu_one_shot_classes_ids, colors, ntu_one_shot_classes):
        #print(i, c, label)
        markerx = "x" if i % 2 else "x"
        #size = size
        #print(label, len(x_reduced[y==i]))
        #print(x_reduced[y==i,0][:])
        plt.scatter(x_reduced[y == i, 0][:], x_reduced[y == i, 1][:],
                    label=label, c=c, s=size, marker=markerx, alpha=1.0)
        #plt.scatter(x_reduced[y == i, 0], x_reduced[y == i, 1], label=label, s=5, marker=markerx, alpha=0.5)
    if legend:
        plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
    plt.savefig(target_filename, bbox_inches='tight')
    #plt.show()


    #scatter = plt.scatter(x_reduced[:num_samples, 0], x_reduced[:num_samples, 1], c=y.flatten(), cmap=plt.cm.get_cmap("Set1", 20),alpha=1.0)
    #scatter = plt.scatter(x_reduced[:num_samples, 0], x_reduced[:num_samples, 1], c=y.flatten(), cmap=plt.cm.get_cmap("rainbow", 27),alpha=0.5)
    #scatter = plt.scatter(x_reduced[:num_samples, 0], x_reduced[:num_samples, 1], c=y.flatten(), cmap=plt.cm.get_cmap("tab20b", len(ntu_one_shot_classes)),alpha=1.0, s=1)
    #plt.legend(handles=scatter.legend_elements()[0], labels=ntu_one_shot_classes)
    #plt.colorbar(ticks=ntu_one_shot_classes_ids)
    #plt.show()
    plt.savefig(target_filename, bbox_inches='tight', dpi=1000)


def plot_embedding(
        filename='/home/kpeng/oneshot_metriclearning/emotic_dml/models/resnet50_emolevel_66_all_re512_stepsize_10_NTU_ONE_SHOT_SIGNALS_model_resnet18_cl_cross_entropy_ml_triplet_margin_miner_multi_similarity_mix_ml_0.50_mix_cl_0.50_resize_256_emb_size_128_class_size_5_opt_rmsprop_lr_0.00_m_0.00_wd_0.00.pkl_last',
        target_filename='tsne.png', mode="tsne", legend=False, size=0.01, color_map="utdmhad"):
    """
    Plots an embedding from a dumped pickel file

    :filename: filename to the pickle file
    :target_filename: output file, different extention fives different format
    :mode: could be either `umap` or `tsne`, depending on your preference
    :legend: if 'True' plots a legend
    """
    print(len(ntu_one_shot_classes))
    with open(filename, "rb") as f:
        a = pickle.load(f)

    num_samples = len(a[0])
    y = a[1][:num_samples].flatten()

    if mode == "tsne":
        #a[0] = (a[0] - a[0].min()) / (a[0].max() - a[0].min())
        data = []
        for i in ntu_one_shot_classes_ids:
            data.append(a[0][y == i][:100])
        data = np.stack(data, axis=1)
        n,w,c = data.shape
        data = np.reshape(data,(n*w, c))
        #kmeanModel = KMeans(n_clusters=6)
        #kmeanModel.fit(data)
        #y = kmeanModel.labels_
        #X_Train_reduced = TruncatedSVD(n_components=6, random_state=0).fit_transform(a[0])
        tsne = TSNE(n_components=2, perplexity=100, early_exaggeration=40, learning_rate=0.01, verbose=2, n_iter=1000)
        x_reduced = tsne.fit_transform(data)
        # x_reduced=pd.DataFrame(tsne.embedding_,index=X_Train_reduced.index)

        # print(x_reduced)

    if mode == "umap":
        x_reduced = umap.UMAP(random_state=42).fit_transform(a[0])
    # print(y)
    print(x_reduced.shape, y.shape)

    # plt.figure(figsize=(, 5))
    plt.rcParams['legend.fontsize'] = 'xx-small'
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    if color_map == "ntu":
        colors = plt.cm.get_cmap("tab20b").colors
    else:
        colors = plt.cm.get_cmap("Set1").colors
    # colors = plt.cm.get_cmap("hsv", 20).colors
    for i, c, label in zip(ntu_one_shot_classes_ids, colors, ntu_one_shot_classes):
        # print(i, c, label)
        markerx = "x" if i % 2 else "x"
        # size = size
        #print(label, len(x_reduced[y == i]))
        # print(x_reduced[y==i,0][:])
        plt.scatter(x_reduced[(i-7)*100:(i-6)*100,0], x_reduced[(i-7)*100:(i-6)*100,1],
                    label=label, c=c, s=size, marker=markerx, alpha=1.0)
        # plt.scatter(x_reduced[y == i, 0], x_reduced[y == i, 1], label=label, s=5, marker=markerx, alpha=0.5)
    if legend:
        plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
    plt.savefig(target_filename, bbox_inches='tight')
    # plt.show()

    # scatter = plt.scatter(x_reduced[:num_samples, 0], x_reduced[:num_samples, 1], c=y.flatten(), cmap=plt.cm.get_cmap("Set1", 20),alpha=1.0)
    # scatter = plt.scatter(x_reduced[:num_samples, 0], x_reduced[:num_samples, 1], c=y.flatten(), cmap=plt.cm.get_cmap("rainbow", 27),alpha=0.5)
    # scatter = plt.scatter(x_reduced[:num_samples, 0], x_reduced[:num_samples, 1], c=y.flatten(), cmap=plt.cm.get_cmap("tab20b", len(ntu_one_shot_classes)),alpha=1.0, s=1)
    # plt.legend(handles=scatter.legend_elements()[0], labels=ntu_one_shot_classes)
    # plt.colorbar(ticks=ntu_one_shot_classes_ids)
    # plt.show()
    plt.savefig(target_filename, bbox_inches='tight', dpi=1000)

def plot_embedding(
        filename='/home/kpeng/oneshot_metriclearning/emotic_dml/models/resnet50_emolevel_66_all_re512_stepsize_10_NTU_ONE_SHOT_SIGNALS_model_resnet18_cl_cross_entropy_ml_triplet_margin_miner_multi_similarity_mix_ml_0.50_mix_cl_0.50_resize_256_emb_size_128_class_size_5_opt_rmsprop_lr_0.00_m_0.00_wd_0.00.pkl_last',
        target_filename='tsne.png', mode="tsne", legend=False, size=0.01, color_map="utdmhad"):
    """
    Plots an embedding from a dumped pickel file

    :filename: filename to the pickle file
    :target_filename: output file, different extention fives different format
    :mode: could be either `umap` or `tsne`, depending on your preference
    :legend: if 'True' plots a legend
    """
    print(len(ntu_one_shot_classes))
    with open(filename, "rb") as f:
        a = pickle.load(f)

    num_samples = len(a[0])
    y = a[1][:num_samples].flatten()

    if mode == "tsne":
        #a[0] = (a[0] - a[0].min()) / (a[0].max() - a[0].min())
        data = []
        for i in ntu_one_shot_classes_ids:
            data.append(a[0][y == i][:100])
        data = np.stack(data, axis=1)
        n,w,c = data.shape
        data = np.reshape(data,(n*w, c))
        #kmeanModel = KMeans(n_clusters=6)
        #kmeanModel.fit(data)
        #y = kmeanModel.labels_
        #X_Train_reduced = TruncatedSVD(n_components=6, random_state=0).fit_transform(a[0])
        tsne = TSNE(n_components=2, perplexity=100, early_exaggeration=40, learning_rate=0.01, verbose=2, n_iter=1000)
        x_reduced = tsne.fit_transform(data)
        # x_reduced=pd.DataFrame(tsne.embedding_,index=X_Train_reduced.index)

        # print(x_reduced)

    if mode == "umap":
        x_reduced = umap.UMAP(random_state=42).fit_transform(a[0])
    # print(y)
    print(x_reduced.shape, y.shape)

    # plt.figure(figsize=(, 5))
    plt.rcParams['legend.fontsize'] = 'xx-small'
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    if color_map == "ntu":
        colors = plt.cm.get_cmap("tab20b").colors
    else:
        colors = plt.cm.get_cmap("Set1").colors
    # colors = plt.cm.get_cmap("hsv", 20).colors
    for i, c, label in zip(ntu_one_shot_classes_ids, colors, ntu_one_shot_classes):
        # print(i, c, label)
        markerx = "x" if i % 2 else "x"
        # size = size
        #print(label, len(x_reduced[y == i]))
        # print(x_reduced[y==i,0][:])
        plt.scatter(x_reduced[(i-7)*100:(i-6)*100,0], x_reduced[(i-7)*100:(i-6)*100,1],
                    label=label, c=c, s=size, marker=markerx, alpha=1.0)
        # plt.scatter(x_reduced[y == i, 0], x_reduced[y == i, 1], label=label, s=5, marker=markerx, alpha=0.5)
    if legend:
        plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
    plt.savefig(target_filename, bbox_inches='tight', dpi=1000)
    # plt.show()

    # scatter = plt.scatter(x_reduced[:num_samples, 0], x_reduced[:num_samples, 1], c=y.flatten(), cmap=plt.cm.get_cmap("Set1", 20),alpha=1.0)
    # scatter = plt.scatter(x_reduced[:num_samples, 0], x_reduced[:num_samples, 1], c=y.flatten(), cmap=plt.cm.get_cmap("rainbow", 27),alpha=0.5)
    # scatter = plt.scatter(x_reduced[:num_samples, 0], x_reduced[:num_samples, 1], c=y.flatten(), cmap=plt.cm.get_cmap("tab20b", len(ntu_one_shot_classes)),alpha=1.0, s=1)
    # plt.legend(handles=scatter.legend_elements()[0], labels=ntu_one_shot_classes)
    # plt.colorbar(ticks=ntu_one_shot_classes_ids)
    # plt.show()
    plt.savefig(target_filename, bbox_inches='tight', dpi=1000)


if __name__ == '__main__':
    plot_embedding_1(filename='/home/kpeng/oneshot_metriclearning/emotic_dml/models/wmo66evel66_fc_lr_stepsize_10_NTU_ONE_SHOT_SIGNALS_model_resnet18_cl_cross_entropy_ml_triplet_margin_miner_multi_similarity_mix_ml_0.50_mix_cl_0.50_resize_256_emb_size_128_class_size_5_opt_rmsprop_lr_0.00_m_0.00_wd_0.00.pkl_last')

from trunk_baseline import Emotic
from pytorch_metric_learning import losses, miners, samplers, trainers, testers, utils
import torch.nn as nn
from emotic_d import Emotic_DataLoader
import record_keeper
import pytorch_metric_learning.utils.logging_presets as logging_presets
from torchvision import datasets, models, transforms
import torchvision
import logging
logging.getLogger().setLevel(logging.INFO)
import os
from scipy.io import loadmat
from torchvision import datasets, transforms
from skimage import io
import os
from pathlib import Path
import numpy as np
import sys
import time
import cv2
from torch.utils.data import Dataset

import pytorch_metric_learning
from pytorch_metric_learning.testers.base_tester import BaseTester
#from resnet import resnet18

logging.info("pytorch-metric-learning VERSION %s"%pytorch_metric_learning.__version__)
logging.info("record_keeper VERSION %s"%record_keeper.__version__)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#from efficientnet_pytorch import EfficientNet
import torch
import numpy as np
import pickle
def recursive_glob(rootdir=".", suffix=".png"):
    return [
        os.path.join(looproot,filename)
        for looproot,_, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def do_knn_and_accuracies(query_embeddings, query_labels, reference_embeddings, reference_labels):
    # print(embeddings_and_labels)
    query_embeddings = np.concatenate(query_embeddings, axis=0)
    #print(query_embeddings.shape)
    query_labels = np.stack(query_labels)
    reference_embeddings = np.concatenate(reference_embeddings,axis=0)
    reference_labels = np.stack(reference_labels)
    # print(reference_labels)
    knn_indices, knn_distances = utils.stat_utils.get_knn(reference_embeddings, query_embeddings, 1, False)
    knn_labels = reference_labels[knn_indices][:, 0]
    #torch.set_printoptions(threshold=10000)
    #np.set_printoptions(threshold=sys.maxsize)

    # print((knn_labels!=7).sum())
    #torch.set_printoptions(threshold=10000)
    #accuracy = accuracy_score(query_labels, knn_labels)
    # print(knn_labels)
    #acc_all = cal_acc(knn_labels.flatten().tolist(), query_labels.flatten().tolist())
    # print(classification_report(knn_labels, query_labels, labels=[7,8,9,10,11,12]))
    # cm = confusion_matrix(knn_labels, query_labels)
    # per_class_accuracies = {}
    # classes = [7,8,9,10,11,12]
    # for idx, cls in enumerate(classes):
    #    # True negatives are all the samples that are not our current GT class (not the current row)
    #    # and were not predicted as the current class (not the current column)
    #    true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
    #
    #    # True positives are all the samples of our current GT class that were predicted as such
    #    true_positives = cm[idx, idx]
    #
    #    # The accuracy for the current class is ratio between correct predictions to all predictions
    #    per_class_accuracies[cls] = (true_positives + true_negatives) / np.sum(cm)
    #print(acc_all)
    #print(accuracy)
    #with open('/home/kpeng/pc14/testing_knn_labels' + "_last", 'wb') as f:
    #    print("Dumping embeddings for new max_acc to file", self.embedding_filename + "_last")
    #    pickle.dump([knn_labels, query_labels, file_list_context], f)
    #accuracies["accuracy"] = accuracy
    #keyname = self.accuracies_keyname("mean_average_precision_at_r")  # accuracy as keyname not working
    #accuracies[keyname] = accuracy
    return knn_labels
def convert_color_to_index(semantic):
    colors = loadmat('/home/kpeng/oneshot_metriclearning/emotic_dml/sl-dml/color150.mat')['colors']
    #print(colors)
    encoder_semantic = torch.zeros(semantic.size()[0],1,150).cuda()
    for i in range(150):
        #print(colors[i][0])
        number = ((semantic[:,0,:,:] == colors[i][0])&(semantic[:,1,:,:] == colors[i][1])&(semantic[:,2,:,:] == colors[i][2])).sum()
        if number != 0:
            encoder_semantic[:,:,i]=1
    return encoder_semantic

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)
def init_weights(m):
    if type(m) == nn.Linear:
        #torch.nn.init.normal(m.weight, 0.1, 0.02)
        torch.nn.init.constant_(m.weight, 0.0)
        m.bias.data.fill_(0.00)
class trunk_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk_context = torchvision.models.resnet18(pretrained=False)
        self.trunk_body = torchvision.models.resnet18(pretrained=False)
        #self.trunk_semantic = torchvision.models.__dict__[cfg.model.model_name](pretrained=False)
        self.trunk_context.fc = Identity()
        self.trunk_body.fc = Identity()
        #self.trunk_semantic.fc = Identity()
        self.embedding = nn.Linear(512+256,512)
        self.semantic = nn.Sequential(nn.Linear(150,256), nn.ReLU(),nn.BatchNorm1d(256), nn.Linear(256, 256))
        #self.semantic.apply(init_weights)
        #self.fc1 = nn.Linear(1000, 512)
        #self.fc2 = nn.Linear(1000, 512)
        self.relu = nn.ReLU()
        #self.embedding_2 = nn.Linear(512,512)
        #self.dropout = nn.Dropout(0.5)
        self.normalize = nn.BatchNorm1d(1000)
        #self.normalize_2 = nn.BatchNorm1d(256)
        self.LSTM = nn.LSTM(1,16,2,bidirectional=True)
        #self.LSTM.weight.data.normal_(0, 0.01)
        #torch.nn.init.xavier_normal(self.LSTM)
        for layer_p in self.LSTM._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    # print(p, a.__getattr__(p))
                    torch.nn.init.normal(self.LSTM.__getattr__(p), 0.4, 0.002) # 0.1 0.002 valence cs 0.0 0.02 dominance csb
                    #torch.nn.init.constant_(self.LSTM.__getattr__(p), 0)
                    # print(p, a.__getattr__(p))
        self.Lift = nn.Linear(150,64)
        self.softmax = nn.Softmax(dim=-1)
        print('successful load trunk2')
    def forward(self,data):
        #print(data)
        #print(label)
        #torch.set_printoptions(threshold=10000)

        data=np.expand_dims(data, axis=0)
        batch_size = data.shape[0]
        data = torch.from_numpy(data).cuda()
        context = data[:,:,:256,:]
        body = data[:,:,256:512,:]
        semantic = data[:,:,512:768,:]
        semantic = convert_color_to_index(semantic)
        semantic = semantic.resize(batch_size, 150)        #sys.exit()
        #print('test')
        feature_context = self.trunk_context(context)
        feature_body =self.trunk_body(body)
        #feature_semantic = self.Lift(torch.mean(self.LSTM(semantic.permute(0,2,1))[0].permute(0,2,1),dim=-2)) #32,150,32
        feature_semantic = self.semantic(semantic)
        #feature_body = self.softmax(feature_semantic)*feature_body
        feature = self.embedding(torch.cat([feature_context,  feature_semantic], dim=1))
        #feature = self.embedding(feature_body)
        #print(feature.size())
        return feature

# This is for replacing the last layer of a pretrained network.
# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def reference_run():
    device = torch.device('cpu')
    emo_dir = 'emotic_cat_oneshot'
    emo_dir_sample = 'emotic_cat_samples'
    cato_list = ['A07', 'A08', 'A09','A10', 'A11','A12']
    load_path_context = '/cvhci/data/activity/kpeng/' + emo_dir + '/context/'
    load_path_body = '/cvhci/data/activity/kpeng/' + emo_dir + '/body/'
    load_path_semantic = '/cvhci/data/activity/kpeng/' + emo_dir + '/semantic/'
    file_list_context, file_list_body, file_list_semantic, targets = [], [], [], []
    for label in cato_list:
        files = recursive_glob(load_path_context + label)
        file_list_context.extend(files)
        targets.extend([int(label.split('A')[-1])] * len(files))
        # print(len(self.file_list_context))
    # print(self.targets)
    for label in cato_list:
        file_list_body.extend(recursive_glob(load_path_body + label))
    for label in cato_list:
        file_list_semantic.extend(recursive_glob(load_path_semantic + label))
    #model_path = "/home/kpeng/oneshot_metriclearning/emotic_dml/sl-dml/outputs/2021-07-05/08-38-00/example_saved_models/resnet50_emolevel_66_cs_re512_stepsize_10_NTU_ONE_SHOT_SIGNALS_model_resnet18_cl_cross_entropy_ml_triplet_margin_miner_multi_similarity_mix_ml_0.50_mix_cl_0.50_resize_256_emb_size_128_class_size_5_opt_rmsprop_lr_0.00_m_0.00_wd_0.00/"
    model_path ="/home/kpeng/oneshot_metriclearning/emotic_dml/sl-dml/outputs/2021-07-05/09-55-03/example_saved_models/resnet50_emolevel_66_base_re512_stepsize_10_NTU_ONE_SHOT_SIGNALS_model_resnet18_cl_cross_entropy_ml_triplet_margin_miner_multi_similarity_mix_ml_0.50_mix_cl_0.50_resize_256_emb_size_128_class_size_5_opt_rmsprop_lr_0.00_m_0.00_wd_0.00/"
    load_path_context_sample = '/cvhci/data/activity/kpeng/' + emo_dir_sample + '/context/'
    load_path_body_sample = '/cvhci/data/activity/kpeng/' + emo_dir_sample + '/body/'
    load_path_semantic_sample = '/cvhci/data/activity/kpeng/' + emo_dir_sample + '/semantic/'
    file_list_context_sample, file_list_body_sample, file_list_semantic_sample, targets_samples = [], [], [],[]
    for label in cato_list:
        files = recursive_glob(load_path_context_sample + label)
        file_list_context_sample.extend(files)
        targets_samples.extend([int(label.split('A')[-1])] * len(files))
        # print(len(self.file_list_context))
    # print(self.targets)
    for label in cato_list:
        file_list_body_sample.extend(recursive_glob(load_path_body_sample + label))
    for label in cato_list:
        file_list_semantic_sample.extend(recursive_glob(load_path_semantic_sample + label))
    trunk = Emotic()
    trunk_output_size = 512


    embedder = MLP([trunk_output_size, 128])
    #classifier = torch.nn.DataParallel(MLP([128, 6])).to(device)

    path_embedder = model_path+ 'embedder_best6.pth'
    #path_classifier = model_path+'classifier_best8.pth'
    path_trunk = model_path+'trunk_best6.pth'
    trunk.load_state_dict(torch.load(path_trunk))
    trunk.eval().cuda()
    embedder.load_state_dict(torch.load(path_embedder))
    embedder.eval().cuda()
    #trunk.load_state_dict(torch.load(path_tr))
    #trunk.eval()
    #load weights
    embedding_samples=[]
    label_samples=[]
    embedding_tests=[]
    label_tests=[]
    sample_range = []
    for i in range(24):
        sample_range.append(range(i*500, (i+1)*500))
    #predictions

    for index in range(len(file_list_context_sample)):
        context_path = file_list_context_sample[index]
        #print(context_path)
        semantic_path = file_list_semantic_sample[index]
        body_path = file_list_body_sample[index]

        context = np.array(io.imread(context_path),dtype=np.float32)

        semantic = np.array(io.imread(semantic_path), dtype=np.float32)
        body = np.array(io.imread(body_path), dtype=np.float32)
        context = np.transpose(cv2.resize(context, [256, 256]),[2,0,1])
        body = np.transpose(cv2.resize(body, [256,256]),[2,0,1])
        semantic = np.transpose(cv2.resize(semantic, [256, 256]),[2,0,1])

        label = int(context_path.split('/')[-2].split('A')[-1])

        data = np.concatenate([context,body,semantic],axis=-2)
        data = torch.from_numpy(data).cuda().unsqueeze(dim=0)
        rep= trunk(data)
        representation_sample = embedder(rep)
        embedding_samples.append(representation_sample.cpu().detach().numpy())
        label_samples.append(label)
        print('samples loading done')
    knn_total=[]
    for sam in sample_range:
        for index in sam:   #(len(file_list_context)):
            context_path = file_list_context[index]
            #print(context_path)
            semantic_path = file_list_semantic[index]
            body_path = file_list_body[index]

            context = np.array(io.imread(context_path),dtype=np.float32)

            semantic = np.array(io.imread(semantic_path), dtype=np.float32)
            body = np.array(io.imread(body_path), dtype=np.float32)
            context = np.transpose(cv2.resize(context, [256, 256]),[2,0,1])
            body = np.transpose(cv2.resize(body, [256,256]),[2,0,1])
            semantic = np.transpose(cv2.resize(semantic, [256, 256]),[2,0,1])

            label = int(context_path.split('/')[-2].split('A')[-1])

            data = np.concatenate([context,body,semantic],axis=-2)
            data = torch.from_numpy(data).cuda().unsqueeze(0)
            rep= trunk(data)
            representation = embedder(rep)
            #print(representation.size())
            #sys.exit()
            embedding_tests.append(representation.cpu().detach().numpy())
            label_tests.append(label)
            print(index)
        knn_indices = do_knn_and_accuracies(embedding_tests, label_tests, embedding_samples, label_samples)
        knn_total.extend(knn_indices)
        embedding_tests=[]
        label_test=[]
    textfile = open("image_list_basebest66.txt", "w")
    for element in file_list_context:
        textfile.write(element + "\n")
    textfile.close()
    np.savetxt('test_base_66.txt', np.stack(knn_total), delimiter=',')  # X is an array

if __name__ == "__main__":
    reference_run()
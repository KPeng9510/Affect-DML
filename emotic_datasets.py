import torch
import cv2
from torchvision import datasets, transforms
from skimage import io
import os
from pathlib import Path
import numpy as np
import sys
import time
from torch.utils.data import Dataset
def recursive_glob(rootdir=".", suffix=".png"):
    return [
        os.path.join(looproot,filename)
        for looproot,_, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]
class Emotic_DataLoader(Dataset):
    def __init__(self, config, mode, emo_mode):
        super(Emotic_DataLoader, self).__init__()
        self._config = config
        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]
        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]
        self.targets = []
        emo_dir = 'emotic_'+str(emo_mode)
        #emo_dir = 'emotic_cat_oneshot'
        if mode == 'Samples':
            emo_dir = 'emotic_sample_'+str(emo_mode)
            #emo_dir = 'emotic_cat_samples'
        if mode == 'Train':
            cato_list = ['A01','A03' ,'A05','A06','A08', 'A09', 'A10']
        elif mode == 'Test':
            cato_list = ['A02','A04', 'A07']
        elif mode == 'Samples':
            cato_list = ['A02','A04' ,'A07']
        
        self.load_path_context = '/cvhci/data/activity/kpeng/'+emo_dir+'/context/'
        self.load_path_body = '/cvhci/data/activity/kpeng/'+emo_dir+'/body/'
        self.load_path_semantic = '/cvhci/data/activity/kpeng/'+emo_dir+'/semantic/'
        self.file_list_context, self.file_list_body,self.file_list_semantic=[],[],[]
        for label in cato_list:
            files = recursive_glob(self.load_path_context+label)
            self.file_list_context.extend(files)
            self.targets.extend([int(label.split('A')[-1])]*len(files))
            #print(len(self.file_list_context))
        #print(self.targets)
        for label in cato_list:
            self.file_list_body.extend(recursive_glob(self.load_path_body+label))
        for label in cato_list:
            self.file_list_semantic.extend(recursive_glob(self.load_path_semantic+label))
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])
        self.resize_context = transforms.Resize((256,256))
        self.resize_semantic = transforms.Resize((256,256))
        self.resize_body = transforms.Resize((256,256))
        #self.train_loader = self._build_train_loader()
        #self.test_loader = self._build_test_loader()

    def __len__(self):
        return len(self.file_list_context)
    def __getitem__(self,index):
        #print(context_path)
        context_path = self.file_list_context[index]
        #print(context_path)
        semantic_path = self.file_list_semantic[index]
        body_path = self.file_list_body[index]
        context = np.array(io.imread(context_path),dtype=np.float32)
        
        semantic = np.array(io.imread(semantic_path), dtype=np.float32)
        body = np.array(io.imread(body_path), dtype=np.float32)
        context = np.transpose(cv2.resize(context, [256, 256]),[2,0,1])
        body = np.transpose(cv2.resize(body, [256,256]),[2,0,1])
        semantic = np.transpose(cv2.resize(semantic, [256, 256]),[2,0,1])
        
        label = int(context_path.split('/')[-2].split('A')[-1])
        
        data = np.concatenate([context,body,semantic],axis=-2)
        #print(data.shape)
        return data,label

    def cat_to_one_hot(self,cat):
        one_hot_cat = np.zeros(10)
        one_hot_cat[label-1]=1
        return one_hot_cat

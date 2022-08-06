import torch 
import torch.nn as nn 
import os
from scipy.io import loadmat
from torch.autograd import Variable as V
import torchvision.models as models
from torch.nn import functional as F

from pathlib import Path

class Emotic(nn.Module):
  ''' Emotic Model'''
  def __init__(self, num_context_features=256, num_body_features=256):
    super(Emotic,self).__init__()
    model_name = 'alexnet_places365.pth.tar'
    model_file = os.path.join(Path('/home/kpeng/oneshot_metriclearning/emotic_dml/resnet_models/'),model_name)
    #print(model_file)
    if not os.path.exists(model_file):
      #print('test')
      download_command = 'wget ' + 'http://places2.csail.mit.edu/models_places365/' + model_name +' -O ' + model_file
      os.system(download_command)
    model_dir = Path('/home/kpeng/oneshot_metriclearning/emotic_dml/resnet_models/')
    save_file = os.path.join(model_dir,'alexnet_places365_py36.pth.tar')
    from functools import partial
    import pickle
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
    torch.save(model, save_file)

    self.model_context = models.__dict__['alexnet'](num_classes=365)
    checkpoint = torch.load(save_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()} # the data parallel layer will add 'module' before each layer name
    self.model_context.load_state_dict(state_dict)
    self.model_context.cuda()
    torch.save(self.model_context, os.path.join(model_dir, 'context_model' + '.pth'))
    self.model_body = models.__dict__['alexnet'](pretrained=True)
    self.model_body.cpu()
    torch.save(self.model_body, os.path.join(model_dir, 'body_model' + '.pth'))
    self.model_body.cuda()
    #self.model_context = nn.Identity()
    #self.model_body = nn.Identity()
    print('finish model preparation!')
    self.num_context_features = num_context_features
    self.num_body_features = num_body_features
    self.fc1 = nn.Linear(1365, 512)
    #self.fc_cont = nn.Linear(365,512)
    #self.fc_body = nn.Linear(1000,512)
    #self.bn1 = nn.BatchNorm1d(256)
    #self.d1 = nn.Dropout(p=0.5)
    #self.fc_cat = nn.Linear(256, 512)
    #self.fc_cont = nn.Linear(256, 3)
    self.relu = nn.ReLU()

    self.LSTM = nn.LSTM(1,16,2,bidirectional=True)
    if 1:
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
  def convert_color_to_index(self, semantic):
      colors = loadmat('/home/kpeng/oneshot_metriclearning/emotic_dml/sl-dml/color150.mat')['colors']
      #print(colors)
      encoder_semantic = torch.zeros(semantic.size()[0],1,150).cuda()
      for i in range(150):
          #print(colors[i][0])
          number = ((semantic[:,0,:,:] == colors[i][0])&(semantic[:,1,:,:] == colors[i][1])&(semantic[:,2,:,:] == colors[i][2])).sum()
          if number != 0:
              encoder_semantic[:,:,i]=1
      return encoder_semantic

  def forward(self, x):
    context = x[:,:,:256,:]
    body = x[:,:,256:512,:]
    semantic = self.convert_color_to_index(x[:,:,512:768,:])
    context_features = self.model_context(context)
    body_features = self.model_body(body)
    feature_semantic = self.Lift(torch.mean(self.LSTM(semantic.permute(0,2,1))[0].permute(0,2,1),dim=-2)) #32,150,32

    #print(context_features.size())
    #print(body_features.size())
    #context_features = x_context.view(-1, 256)
    #body_features = x_body.view(-1, 256)
    fuse_features = torch.cat((context_features, body_features), 1)
    fuse_out = self.fc1(fuse_features)
    #fuse_out = self.bn1(fuse_out)
    #fuse_out = self.relu(fuse_out)
    #fuse_out = self.d1(fuse_out)    
    #cat_out = self.fc_cat(fuse_out)
    #cont_out = self.fc_cont(fuse_out)
    return fuse_out

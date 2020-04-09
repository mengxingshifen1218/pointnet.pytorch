from __future__ import print_function
# import os
# import sys
# print(sys.path)
# sys.path.append(os.path.dirname(os.getcwd()))
# print(sys.path)

import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# from utils.train_classification import logger

# INFO CRITICAL
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s') 


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x): # torch.Size([2, 3, 2500])
        batchsize = x.size()[0]  # num is:2

        logging.info("The x type is:%s,shape is:%s",str(type(x)),str(x.shape))  
        logging.info("The batchsize type is:%s,num is:%d",str(type(batchsize)),batchsize) 

        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([2, 64, 2500])
        logging.info("The x1 type is:%s,shape is:%s",str(type(x)),str(x.shape))  

        x = F.relu(self.bn2(self.conv2(x))) # torch.Size([2, 128, 2500])
        logging.info("The x2 type is:%s,shape is:%s",str(type(x)),str(x.shape))  

        x = F.relu(self.bn3(self.conv3(x))) # torch.Size([2, 1024, 2500])
        logging.info("The x3 type is:%s,shape is:%s",str(type(x)),str(x.shape))  

        x = torch.max(x, 2, keepdim=True)[0] # Size([2, 1024, 1])
        logging.info("The x4 type is:%s,shape is:%s",str(type(x)),str(x.shape))  

        x = x.view(-1, 1024) # torch.Size([2, 1024])
        logging.info("The x5 type is:%s,shape is:%s",str(type(x)),str(x.shape))  


        x = F.relu(self.bn4(self.fc1(x))) # torch.Size([2, 512])
        logging.info("The x6 type is:%s,shape is:%s",str(type(x)),str(x.shape))  

        x = F.relu(self.bn5(self.fc2(x))) # torch.Size([2, 256])
        logging.info("The x7 type is:%s,shape is:%s",str(type(x)),str(x.shape))  

        x = self.fc3(x) # torch.Size([2, 9])
        logging.info("The x8 type is:%s,shape is:%s",str(type(x)),str(x.shape))  

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        # torch.Size([2, 9])
        # tensor([[1., 0., 0., 0., 1., 0., 0., 0., 1.],[1., 0., 0., 0., 1., 0., 0., 0., 1.]])

        logging.info("The iden type is:%s,shape is:%s",str(type(iden)),str(iden.shape))  
        # logging.info(iden)  
        
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3) # torch.Size([2, 3, 3])

        logging.info("The x9 type is:%s,shape is:%s",str(type(x)),str(x.shape)) 

        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x): # torch.Size([2, 64, 2500])
        logging.info("The x0 type is:%s,shape is:%s",str(type(x)),str(x.shape)) 

        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x))) # torch.Size([2, 64, 2500])
        logging.info("The x1 type is:%s,shape is:%s",str(type(x)),str(x.shape)) 

        x = F.relu(self.bn2(self.conv2(x))) # torch.Size([2, 128, 2500])
        logging.info("The x2 type is:%s,shape is:%s",str(type(x)),str(x.shape)) 

        x = F.relu(self.bn3(self.conv3(x))) # torch.Size([2, 1024, 2500])
        logging.info("The x3 type is:%s,shape is:%s",str(type(x)),str(x.shape)) 

        x = torch.max(x, 2, keepdim=True)[0] # torch.Size([2, 1024, 1])
        logging.info("The x4 type is:%s,shape is:%s",str(type(x)),str(x.shape)) 

        x = x.view(-1, 1024) # torch.Size([2, 1024])
        logging.info("The x5 type is:%s,shape is:%s",str(type(x)),str(x.shape)) 


        x = F.relu(self.bn4(self.fc1(x))) # torch.Size([2, 512])
        logging.info("The x6 type is:%s,shape is:%s",str(type(x)),str(x.shape)) 

        x = F.relu(self.bn5(self.fc2(x))) # torch.Size([2, 256])
        logging.info("The x7 type is:%s,shape is:%s",str(type(x)),str(x.shape)) 

        x = self.fc3(x) # torch.Size([2, 4096])
        logging.info("The x8 type is:%s,shape is:%s",str(type(x)),str(x.shape)) 

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        logging.info("The iden type is:%s,shape is:%s",str(type(iden)),str(iden.shape))  
        logging.info(iden)  

        # iden torch.Size([2, 4096])

        if x.is_cuda:
            iden = iden.cuda()

        x = x + iden
        x = x.view(-1, self.k, self.k) # torch.Size([2, 64, 64])
        logging.info("The x9 type is:%s,shape is:%s",str(type(x)),str(x.shape))

        # input('please pause')
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):  
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x): # [2,3,2500]
        n_pts = x.size()[2] # <class 'int'>,shape is:2500
        logging.info("The n_pts type is:%s,shape is:%d",str(type(n_pts)),n_pts) 
        
        trans = self.stn(x) # torch.Size([2, 3, 3])
        logging.info("The trans type is:%s,shape is:%s",str(type(trans)),str(trans.shape))

        x = x.transpose(2, 1) # torch.Size([2, 2500, 3])
        logging.info("The x1 type is:%s,shape is:%s",str(type(x)),str(x.shape))
        
        x = torch.bmm(x, trans) # torch.Size([2, 2500, 3])
        logging.info("The x2 type is:%s,shape is:%s",str(type(x)),str(x.shape))

        x = x.transpose(2, 1) # torch.Size([2, 3, 2500])
        logging.info("The x3 type is:%s,shape is:%s",str(type(x)),str(x.shape))

        x = F.relu(self.bn1(self.conv1(x)))  # torch.Size([2, 64, 2500])
        logging.info("The x4 type is:%s,shape is:%s",str(type(x)),str(x.shape))

        if self.feature_transform: # if true then use 
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x # torch.Size([2, 64, 2500])

        x = F.relu(self.bn2(self.conv2(x))) # torch.Size([2, 128, 2500])
        logging.info("The x5 type is:%s,shape is:%s",str(type(x)),str(x.shape))

        x = self.bn3(self.conv3(x)) # torch.Size([2, 1024, 2500])
        logging.info("The x6 type is:%s,shape is:%s",str(type(x)),str(x.shape))

        x = torch.max(x, 2, keepdim=True)[0] #t orch.Size([2, 1024, 1])
        logging.info("The x7 type is:%s,shape is:%s",str(type(x)),str(x.shape))

        x = x.view(-1, 1024) # torch.Size([2, 1024])
        logging.info("The x8 type is:%s,shape is:%s",str(type(x)),str(x.shape))

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts) # torch.Size([2, 1024, 2500])
            return torch.cat([x, pointfeat], 1), trans, trans_feat
            # torch.cat([x, pointfeat], 1) torch.Size([2, 1088, 2500])

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        # x      torch.Size([2, 1024])
        # trans  torch.Size([2, 3, 3])
        # trans_feat ： NONE or torch.Size([2, 64, 64])

        logging.info("The x type is:%s,shape is:%s",str(type(x)),str(x.shape))
        logging.info("The trans type is:%s,shape is:%s",str(type(trans)),str(trans.shape))
        # logging.info("The trans_feat type is:%s,shape is:%s",str(type(trans_feat)),str(trans_feat.shape))
        

        x = F.relu(self.bn1(self.fc1(x))) # torch.Size([2, 512])
        logging.info("The x1 type is:%s,shape is:%s",str(type(x)),str(x.shape))

        x = F.relu(self.bn2(self.dropout(self.fc2(x)))) # torch.Size([2, 256])
        logging.info("The x2 type is:%s,shape is:%s",str(type(x)),str(x.shape))

        x = self.fc3(x) # torch.Size([2, 16])
        logging.info("The x3 type is:%s,shape is:%s",str(type(x)),str(x.shape))

        return F.log_softmax(x, dim=1), trans, trans_feat
        # F.log_softmax(x, dim=1) shape: torch.Size([2, 16])

class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x): # [2,3,2500]

        batchsize = x.size()[0]

        n_pts = x.size()[2]

        x, trans, trans_feat = self.feat(x)  # torch.Size([2, 1088, 2500])
        

        x = F.relu(self.bn1(self.conv1(x)))  # torch.Size([2, 512, 2500])

        x = F.relu(self.bn2(self.conv2(x)))  # torch.Size([2, 256, 2500])

        x = F.relu(self.bn3(self.conv3(x)))  # torch.Size([2, 128, 2500])

        x = self.conv4(x)                    # torch.Size([2, k, 2500])

        x = x.transpose(2,1).contiguous()    # torch.Size([2, 2500, k])

        x = F.log_softmax(x.view(-1,self.k), dim=-1)  # torch.Size([5000, 10])

        x = x.view(batchsize, n_pts, self.k) # torch.Size([2, 2500, k])

        return x, trans, trans_feat

def feature_transform_regularizer(trans): # torch.Size([2, 64, 64])

    logging.info("The trans type is:%s,shape is:%s",str(type(trans)),str(trans.shape))

    d = trans.size()[1]
    logging.info("The d num is:%d",d)

    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :] # torch.Size([1, 64, 64])
    logging.info("The I type is:%s,shape is:%s",str(type(I)),str(I.shape))

    if trans.is_cuda:
        I = I.cuda()
    
    # logging.info(trans.transpose(2,1).shape) # torch.Size([2, 64, 64])
    # logging.info(torch.bmm(trans, trans.transpose(2,1)).shape) # torch.Size([2, 64, 64])
    # logging.info(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)).shape) # torch.Size([2])
    # logging.info(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2))) # tensor([40.8016, 52.1468], device='cuda:0', grad_fn=<SqrtBackward>)
   

    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))

    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())

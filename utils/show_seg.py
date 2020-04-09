from __future__ import print_function
import os
import sys
print(sys.path)
sys.path.append(os.path.dirname(os.getcwd()))
print(sys.path)

from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
#parser.add_argument('--dataset', type=str, default='', help='dataset path')
parser.add_argument('--dataset', type=str, default='/home/djq/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0', help="dataset path")
parser.add_argument('--class_choice', type=str, default='', help='class choice')

# 输出一行状态栏参数如下:
# 				Namespace(class_choice='Airplane', dataset='', 
# 							idx=2, model='seg/seg_model_Chair_1.pth')

opt = parser.parse_args()
print(opt)

d = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)

idx = opt.idx

# print("d:{}".format(d))
print("model %d/%d" % (idx, len(d))) # d代表全部的飞机数量
# model 2/341


point, seg = d[idx] # 模型里的第idx的点云
print(point.size(), seg.size()) # seg代表每一个点的标签 
# torch.Size([2500, 3]) torch.Size([2500])

point_np = point.numpy() #将torch转为numpy

# 可视化
cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy() - 1, :]

# 载入模型
state_dict = torch.load(opt.model)
classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0])
classifier.load_state_dict(state_dict)
classifier.eval() #评估

# 点云转置
point = point.transpose(1, 0).contiguous()
print('point.transpose(1, 0).shape: ',point.shape)

point = Variable(point.view(1, point.size()[0], point.size()[1]))
pred, _, _ = classifier(point) #分割
# print('\npred.shape:',pred[0],'\n')
pred_choice = pred.data.max(2)[1]
print(pred_choice) #输出每一个点的预测类别

#print(pred_choice.size())
print(pred_choice.numpy()[0])  #[1 1 1 ... 1 1 1]
pred_color = cmap[pred_choice.numpy()[0], :] #根据分类结果显示颜色
print('\npred_color: ',pred_color.shape,'\n')

#print(pred_color.shape)
showpoints(point_np, gt, pred_color) #pred_colord的为(2500, 3)的矩阵
print(point_np.shape)

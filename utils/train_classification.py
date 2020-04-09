from __future__ import print_function
import os
import sys
print(sys.path)
sys.path.append(os.path.dirname(os.getcwd()))
print(sys.path)

import logging  # 引入logging模块
import time
import argparse
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size') #32
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')

parser.add_argument('--dataset', type=str, default='/home/djq/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0', help="dataset path")
#parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

f = open(sys.path[0] + '/test.txt', 'w')
f.truncate() # 清空txt
f.close()

#--------------------------------------------------------------#
# 第一步，创建一个logger
logger = logging.getLogger('test') #设置logger 记录器 # 可以自己定义名字
logger.setLevel(logging.INFO) #Log等级总开关

# 第二步，创建一个handler，用于写入日志文件
# rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
# log_path = os.path.dirname(os.getcwd()) + '/Logs/'
# log_name = log_path + rq + '.log'
# logfile = log_name
file_handler = logging.FileHandler(filename='test.txt')
file_handler.setLevel(logging.INFO)  # 输出到file的log等级的开关 

# 创建一个handler，用于输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 输出到console的log等级的开关  CRITICAL

# file_handler = logging.FileHandler(str(log_dir) + '/train_%s_partseg.txt'%args.model_name) #设置handler处理器 输出目录
# file_handler.setLevel(logging.INFO) # 输出到file的log等级的开关

# 第三步，定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')#设置输出的布局
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 第四步，将handler添加到logger里面
logger.addHandler(file_handler)
logger.addHandler(console_handler)
#--------------------------------------------------------------#

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):

    scheduler.step()

    for i, data in enumerate(dataloader, 0):

        points, target = data # The data type is:<class 'list'>
        logger.info("The data type is:%s",str(type(data)))
        logger.info("The points type is:%s,shape is:%s",str(type(points)),str(points.shape)) # torch.Size([2, 2500, 3]
        logger.info("The target type is:%s,shape is:%s",str(type(target)),str(target.shape)) # torch.Size([2, 1])
        logger.info(target) # tensor([[ 4],[11]])


        target = target[:, 0]
        logger.info("The target type is:%s,shape is:%s",str(type(target)),str(target.shape)) # torch.Size([2])
        logger.info(target) # tensor([ 4, 11])

        points = points.transpose(2, 1)
        logger.info("The points type is:%s,shape is:%s",str(type(points)),str(points.shape)) # torch.Size([2, 3, 2500])

        points, target = points.cuda(), target.cuda()


        optimizer.zero_grad()

        classifier = classifier.train()

        pred, trans, trans_feat = classifier(points)

        logger.info("The pred type is:%s,shape is:%s",str(type(pred)),str(pred.shape))  # <class 'torch.Tensor'>,shape is:torch.Size([2, 16])
        logger.info("The trans type is:%s,shape is:%s",str(type(trans)),str(trans.shape))  # <class 'torch.Tensor'>,shape is:torch.Size([2, 3, 3])
        logger.info("The trans_feat type is:%s",str(type(trans_feat))) # <class 'NoneType'> or torch.Size([2, 64, 64])

        # logger.info("The pred type is:%s",str(type(pred)))
        # logger.info("The trans type is:%s",str(type(trans))) 
        # logger.info("The trans_feat type is:%s",str(type(trans_feat)))

        loss = F.nll_loss(pred, target) # torch.Size([])  tensor(2.6180)
        logger.info("The loss type is:%s,shape is:%s",str(type(loss)),str(loss.shape)) 
        # logger.info(loss)


        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        loss.backward()

        optimizer.step()

        pred_choice = pred.data.max(1)[1] # tensor([ 6, 13], device='cuda:0') softmax 最大的label

        # logger.info("----------------------------------------")
        # logger.info(pred.shape)
        # logger.info(pred.data.shape)
        # logger.info(pred)
        # logger.info(pred.data)
        # logger.info(pred.data.max(1)) # tensor([-2.2310, -2.4911], device='cuda:0'), tensor([10,  4], device='cuda:0')
        # logger.info(pred.data.max(1)[1]) # tensor([ 6, 13], device='cuda:0')

        correct = pred_choice.eq(target.data).cpu().sum()
        
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

# input('please pause')

total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))


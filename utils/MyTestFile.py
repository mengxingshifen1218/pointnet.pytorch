#!/usr/local/bin/python
# -*- coding:utf-8 -*-
import logging  # 引入logging模块
import os.path
import time
import sys
import torch
import torch.nn.functional as F

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
# # 由于日志基本配置中级别设置为DEBUG，所以一下打印信息将会全部显示在控制台上
# logging.info('this is a loggging info message')
# logging.debug('this is a loggging debug message')
# logging.warning('this is loggging a warning message')
# logging.error('this is an loggging error message')
# logging.critical('this is a loggging critical message')

# 第一步，创建一个logger
logger = logging.getLogger('test') #设置logger 记录器 # 可以自己定义名字
logger.setLevel(logging.INFO) #Log等级总开关

# 第二步，创建一个handler，用于写入日志文件
# rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
# log_path = os.path.dirname(os.getcwd()) + '/Logs/'
# log_name = log_path + rq + '.log'
# logfile = log_name

# rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
# log_path = os.path.dirname(os.getcwd()) + '-Logs-'
# rn = '-train_test.txt'
# logfile = log_path + rq + rn
  
# file_handler = logging.FileHandler(filename=logfile)

file_handler = logging.FileHandler(filename='example.log')
file_handler.setLevel(logging.INFO)  # 输出到file的log等级的开关

# 创建一个handler，用于输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.CRITICAL)  # 输出到console的log等级的开关

# file_handler = logging.FileHandler(str(log_dir) + '/train_%s_partseg.txt'%args.model_name) #设置handler处理器 输出目录
# file_handler.setLevel(logging.INFO) # 输出到file的log等级的开关

# 第三步，定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')#设置输出的布局
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 第四步，将handler添加到logger里面
logger.addHandler(file_handler)
logger.addHandler(console_handler)

x = time.localtime(time.time())

logger.info('---------------------------------------------------TRANING---------------------------------------------------')
logger.info('PARAMETER ...')

logger.debug('this is a logger debug message')
logger.info('this is a logger info message')
logger.warning('this is a logger warning message')
logger.error('this is a logger error message')
logger.critical('this is a logger critical message')
logger.critical(type(os.path.dirname(os.getcwd())))
logger.critical(os.path.dirname(os.getcwd()))
logger.critical(os.path)
logger.critical(sys.path)


# os.path.dirname(os.getcwd()) + '/train_%s_partseg.txt'%args.model_name

a = torch.rand(2,3,4)
print(a.shape)
print(a)
b = a.view(-1,4)
print(b.shape)
print(b)

c = F.log_softmax(b.view(-1,4), dim=-1)
print(c.shape)
print(c)

d = c.view(2,3,4)
print(d.shape)
print(d)

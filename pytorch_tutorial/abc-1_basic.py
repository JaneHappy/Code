# coding: utf-8
# Basic PyTorch

from __future__ import division
from __future__ import print_function

import numpy as np 

import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 




'''
class Net(nn.Module):
	"""docstring for Net"""
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input image channel, 6 output channels, 5x5 square convolution kernel
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)  # an affine operation: y = Wx+b
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
'''




# 基本的网络构建类模板
class net_name(nn.Module):
	"""docstring for net_name"""
	def __init__(self):
		super(net_name, self).__init__()
		# 可以添加各种网络层
		self.conv1 = nn.Conv2d(3, 10, 3)  # 1 input image channel, 10 output channels, 3x3 square convolution kernel
		# 具体每种层的参数可以去查看文档

	def forward(self, x):
		# 定义向前传播
		out = self.conv1(x)
		return out










'''
References:
https://www.jianshu.com/p/a9571c537b59
http://www.pytorchtutorial.com/10-minute-pytorch-0/
'''

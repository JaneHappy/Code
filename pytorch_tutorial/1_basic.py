# coding: utf-8
# Basic PyTorch

from __future__ import division
from __future__ import print_function

import numpy as np 

import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 




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
References:
https://www.jianshu.com/p/a9571c537b59
'''

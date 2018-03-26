# coding: utf-8
# Reference:
# https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch/tree/master/chapter2_PyTorch-Basics




from __future__ import division
from __future__ import print_function

import torch
import numpy as np 




'''

# Create a ``numpy ndarray''
np_tensor = np.random.randn(10, 20)

# ->  PyTorch tensor
pt_tensor1 = torch.Tensor(np_tensor)
pt_tensor2 = torch.from_numpy(np_tensor)

# ->  numpy ndarray
np_ts = pt_tensor1.numpy() 			#if it's on CPU
np_ts = pt_tensor1.cpu().numpy() 	#if it's on GPU
# 需要注意 GPU 上的 Tensor 不能直接转换为 NumPy ndarray，需要使用.cpu()先将 GPU 上的 Tensor 转到 CPU 上

# Using GPU
#PyTorch Tensor 使用 GPU 加速
# 第一种方式是定义 cuda 数据类型
dtype = torch.cuda.FloatTensor 				# 定义默认 GPU 的 数据类型
gpu_tensor = torch.randn(10, 20).type(dtype)
# 第二种方式更简单，推荐使用
gpu_tensor = torch.randn(10, 20).cuda(0) 	# 将 tensor 放到第一个 GPU 上
gpu_tensor = torch.randn(10, 20).cuda(1) 	# 将 tensor 放到第二个 GPU 上
# 使用第一种方式将 tensor 放到 GPU 上的时候会将数据类型转换成定义的类型，而是用第二种方式能够直接将 tensor 放到 GPU 上，类型跟之前保持一致
# 推荐在定义 tensor 的时候就明确数据类型，然后直接使用第二种方法将 tensor 放到 GPU 上

# 而将 tensor 放回 CPU 的操作非常简单
cpu_tensor = gpu_tensor.cpu()

'''




from torch.autograd import Variable









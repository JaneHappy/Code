# coding: utf-8
# GAN, vae




import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F 
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as tfs 
from torchvision.utils import save_image

im_tfs = tfs.Compose([
	tfs.ToTensor(), 
	tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 	# 标准化
])
train_set  = MNIST('./torch_mnist', transform=im_tfs)
train_data = DataLoader(train_set, batch_size=128, shuffle=True)


class VAE(nn.Module):
	"""docstring for VAE"""
	def __init__(self):
		super(VAE, self).__init__()
		
		self.fc1  = nn.Linear(784, 400)
		self.fc21 = nn.Linear(400, 20) 	# mean
		self.fc22 = nn.Linear(400, 20) 	# var
		self.fc3  = nn.Linear(20, 400)
		self.fc4  = nn.Linear(400, 784)

	def encode(self, x):
		h1 = F.relu(self.fc1(x))
		return self.fc21(h1), self.fc22(h1)

	def reparametrize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		eps = torch.FloatTensor(std.size()).normal_()
		if torch.cuda.is_available():
			eps = Variable(eps.cuda())
		else:
			eps = Variable(eps)
		return eps.mul(std).add_(mu)

	def decode(self, z):
		h3 = F.relu(self.fc3(z))
		return F.tanh(self.fc4(h3))

	def forward(self, x):
		mu, logvar = self.encode(x) 	# 编码
		z = self.reparametrize(mu, logvar) 	# 重新参数化成正态分布
		return self.decode(z), mu, logvar 	# 解码，同时输出均值方差


net = VAE() 	# 实例化网络
if torch.cuda.is_available():
	net = net.cuda()

x, _ = train_set[0]
x = x.view(x.shape[0], -1)
if torch.cuda.is_available():
	x = x.cuda()
x = Variable(x)
_, mu, var = net(x)

print(mu)
# 可以看到，对于输入，网络可以输出隐含变量的均值和方差，这里的均值方差还没有训练
'''
Variable containing:
	Columns 0 to 9 
		 0.2432  0.0203  0.0313 -0.3649  0.4421 -0.1722  0.4230 -0.0275 -0.2019 -0.2072
	Columns 10 to 19 
		 0.7183 -0.0065  0.1228 -0.4835  0.1444  0.2287  0.2773 -0.0295  0.5154  0.4420
[torch.FloatTensor of size 1x20]
[Finished in 2.2s]
'''



# 下面开始训练
reconstruction_function = nn.MSELoss(size_average=False)

def loss_function(recon_x, x, mu, logvar):
	"""
	recon_x :	generating images
	x 		:	origin images
	mu 		:	latent mean
	logvar 	:	latent log variance
	"""
	MSE = reconstruction_function(recon_x, x)
	# loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
	'''
	(mu^2 + exp(logvar)) * (-1) + 1 + logvar
	'''
	KLD = torch.sum(KLD_element).mul_(-0.5)
	# KL divergence
	return MSE + KLD

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

def to_img(x):
	"""
	定义一个函数将最后的结果转换回图片
	"""
	x = 0.5 * (x + 1.)
	x = x.clamp(0, 1)
	x = x.view(x.shape[0], 1, 28, 28)
	return x


for e in range(40): #100
	for im, _ in train_data:
		im = im.view(im.shape[0], -1)
		im = Variable(im)
		if torch.cuda.is_available():
			im = im.cuda()
		recon_im, mu, logvar = net(im)
		loss = loss_function(recon_im, im, mu, logvar) / im.shape[0] 	# 将 loss 平均
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if (e+1) % 20 == 0:
		print('epoch: {}, Loss: {:.4f}'.format(e + 1, loss.data[0] ))
		save = to_img(recon_im.cpu().data)
		if not os.path.exists('./vae_img'):
			os.mkdir('./vae_img')
		save_image(save, './vae_img/image_{}.png'.format(e + 1))


# save
torch.save(net.state_dict(), './vae_img/net_params.pkl')

# 我们可以输出其中的均值看看
x, _ = train_set[0]
x = x.view(x.shape[0], -1)
if torch.cuda.is_available():
	x = x.cuda()
x = Variable(x)
_, mu, _ = net(x)
print(mu)

'''
RuntimeError: Dataset not found. You can use download=True to download it
(myvenv) byj@ubri07:~/temporal$ python ch6_GAN.py
(1L, 3L)
epoch: 20, Loss: 102.7229
epoch: 40, Loss: 98.3499
epoch: 60, Loss: 91.1938
epoch: 80, Loss: 89.9448
epoch: 100, Loss: 99.5597
(myvenv) byj@ubri07:~/temporal$ python ch6_GAN.py
(1L, 3L)
epoch: 20, Loss: 96.5108
epoch: 40, Loss: 93.9272
(myvenv) byj@ubri07:~/temporal$

(myvenv) byj@ubri07:~/temporal$ python ch6-2_GAN_vae.py
Variable containing:
	Columns 0 to 9
		-0.0803 -0.0533 -0.2055 -0.3762  0.0036 -0.1733 -0.1305  0.2686  0.1381  0.0204
	Columns 10 to 19
		-0.1863 -0.4373 -0.2761  0.0064 -0.0954 -0.3002 -0.1509 -0.1063 -0.0506 -0.1334
[torch.cuda.FloatTensor of size 1x20 (GPU 0)]
epoch: 20, Loss: 67.0201
epoch: 40, Loss: 61.0904
Variable containing:
	Columns 0 to 9
		 0.9836 -1.3679  0.2100  0.5428 -1.2871 -0.5007 -0.0266 -0.4673  2.5180 -1.6126
	Columns 10 to 19
		 0.6913  1.9334  1.0419 -1.5030 -1.9046  0.7899 -1.0324 -0.4611  0.4471  0.6330
[torch.cuda.FloatTensor of size 1x20 (GPU 0)]
(myvenv) byj@ubri07:~/temporal$
'''




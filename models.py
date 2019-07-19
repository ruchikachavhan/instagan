import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets 
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import os
from itertools import zip_longest as zip
from itertools import chain
import numpy as np


class ResidualBlock(nn.Module):
	def __init__(self):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 0)
	def forward(self, x):
		for i in range(0, 6):
			x = self.conv1(self.conv1(x))
		return x

class DBlock(nn.Module):
	def __init__(self):
		super(DBlock, self).__init__()
		self.conv128 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 4)
		self.conv256 = nn.Conv2d(128, 256, kernel_size = 3, stride= 1, padding = 4)
		self.instnorm1 = nn.InstanceNorm2d(128)
		self.instnorm2 = nn.InstanceNorm2d(256)
	def forward(self, x):
		conv_1 = F.relu(self.instnorm1(self.conv128(x)))
		conv_2 = F.relu(self.instnorm2(self.conv256(conv_1)))
		return conv_2

class UBlock(nn.Module):
	def __init__(self, num_channels):
		super(UBlock, self).__init__()
		self.u128 = nn.ConvTranspose2d(num_channels, 128, kernel_size = 3, stride = 1, padding =0)
		self.u64 = nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 1, padding =0)
		self.instnorm1 = nn.InstanceNorm2d(128)
		self.instnorm2 = nn.InstanceNorm2d(64)
	def forward(self, x):
		conv_1 = F.relu(self.instnorm1(self.u128(x)))
		conv_2 = F.relu(self.instnorm2(self.u64(conv_1)))
		return conv_2
  
  
class FeatureExtractor(nn.Module):
  def __init__(self, num_channels):
    super(FeatureExtractor, self).__init__()
    self.conv7s1_64 = nn.Conv2d(num_channels, 64, kernel_size = 7, stride = 1,padding = 4)
    self.R_Block = ResidualBlock()
    self.D_Block = DBlock()
  def forward(self, x):
    x = self.conv7s1_64(x)
    x = self.D_Block.forward(x)
    x = self.R_Block.forward(x)
    return x 

  
class FeatureGeneratorMask(nn.Module):
  def __init__(self):
    super(FeatureGeneratorMask, self).__init__()
    self.conv7s1_3 = nn.Conv2d(64, 1, kernel_size= 7, stride = 1, padding = 6)
    self.U_block = UBlock(768)
  def forward(self, x):
    x = self.U_block.forward(x)
    x = F.relu(self.conv7s1_3(x))
    return x
  
class FeatureGenerator(nn.Module):
  def __init__(self):
    super(FeatureGenerator, self).__init__()
    self.conv7s1_3 = nn.Conv2d(64, 3, kernel_size= 7, stride = 1, padding = 6)
    self.U_block = UBlock(512)
  def forward(self, x):
    x = self.U_block.forward(x)
    x = F.relu(self.conv7s1_3(x))
    return x

class ResnetGenerator(nn.Module):
  def __init__(self):
    super(ResnetGenerator, self).__init__()
    self.Feature_x_mask = FeatureExtractor(1)
    self.Feature_x_orig = FeatureExtractor(3)
    self.Feature_y_mask = FeatureExtractor(1)
    self.Feature_y_orig = FeatureExtractor(3)
    self.FeatureG_x_mask = FeatureGeneratorMask()
    self.FeatureG_x_orig = FeatureGenerator()
    self.FeatureG_y_mask = FeatureGeneratorMask()
    self.FeatureG_y_orig = FeatureGenerator()
  def forward(self, images, images_orig):
    features_mask = self.Feature_x_mask.forward(images)
    features_orig = self.Feature_x_orig.forward(images_orig)
    out = torch.cat((features_orig, features_mask,  features_mask), dim = 1)
    out_orig = torch.cat((features_orig, features_mask), dim = 1)   
    out_g = self.FeatureG_x_mask.forward(out)
    out_g_orig = self.FeatureG_x_orig.forward(out_orig)
    return out_g, out_g_orig
     

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.conv1 =  nn.utils.spectral_norm(nn.Conv2d(4, 64, kernel_size = 4, stride = 2, padding = 0))
		self.conv2 =  nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 0))
		self.conv3 =  nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 0))
		self.conv4 =  nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 0))
		self.conv5 =  nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size = 2, stride = 1, padding =0))
		self.relu = nn.LeakyReLU(0.2)
	def forward(self, x):
		conv_1 = self.relu(self.conv1(x))
		conv_2 = self.relu(self.conv2(conv_1))
		conv_3 = self.relu(self.conv3(conv_2))
		conv_4 = self.relu(self.conv4(conv_3))
		out = torch.sigmoid((self.conv5(conv_4)))
		return out

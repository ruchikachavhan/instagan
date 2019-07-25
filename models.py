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
  def __init__(self, in_features):
    super(ResidualBlock, self).__init__()

    conv_block = [  nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features, in_features, 3),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features, in_features, 3),
                    nn.InstanceNorm2d(in_features)  ]

    self.conv_block = nn.Sequential(*conv_block)

  def forward(self, x):
    return x + self.conv_block(x)
  
class FeatureExtractor(nn.Module):
  def __init__(self, input_nc):
    super(FeatureExtractor, self).__init__()
    model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
    in_features = 64
    out_features = in_features*2
    for _ in range(2):
        model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) ]
        in_features = out_features
        out_features = in_features*2

    # Residual blocks
    for _ in range(6):
        model += [ResidualBlock(in_features)]
    self.model = nn.Sequential(*model)
  def forward(self, x):
    return self.model(x) 

  
class FeatureGeneratorMask(nn.Module):
  def __init__(self):
    super(FeatureGeneratorMask, self).__init__()
    in_features = 768
    out_features = 128
    model = [  nn.ConvTranspose2d(768, 128, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) ]
    model += [  nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True) ]


    # Output layer
    model += [  nn.ReflectionPad2d(3),
                nn.Conv2d(64, 1, 7) ]

    self.model = nn.Sequential(*model)

  def forward(self, x):
      return self.model(x)
  
class FeatureGenerator(nn.Module):
  def __init__(self):
    super(FeatureGenerator, self).__init__()
    in_features = 512
    out_features = 128
    model = [  nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) ]
    model += [  nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True) ]


    # Output layer
    model += [  nn.ReflectionPad2d(3),
                nn.Conv2d(64, 3, 7) ]

    self.model = nn.Sequential(*model)

  def forward(self, x):
      return self.model(x)


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

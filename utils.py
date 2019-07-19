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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

def return_cuda_tensor(t):
	t= t.float()
	t = t.to(device)
	return t

def Sample_images(list):
  if(len(list)>50):
    index = len(list) - 50
    list = list[index:len(list)]
  index = np.random.randint(0, len(list))
  return list[index]

def get_weights(x, y):
  weight = torch.zeros(x.size())
  for i in range (0, x.shape[2]):
    for j in range(0, x.shape[3]):
      weight[0][0][i][j] = 1.00 - min(x[0][0][i][j], y[0][0][i][j])
  #weight = torch.cuda.FloatTensor(weight)
  return weight

def calc_gen_loss(images_x, images_orig_x,images_y, images_orig_y):
#LSGAN loss
  G_xy.zero_grad()
  G_yx.zero_grad()

  x_id, x_id_orig = G_yx.forward(images_x, images_orig_x)
  y_id, y_id_orig = G_xy.forward(images_y, images_orig_y)

  y_fake, y_fake_orig = G_xy.forward(images_x, images_orig_x)
  x_fake, x_fake_orig = G_yx.forward(images_y, images_orig_y)

  re_x, re_x_orig = G_yx.forward(y_fake, y_fake_orig)
  re_y, re_y_orig = G_xy.forward(x_fake, x_fake_orig)

  x_concat = torch.cat(( x_fake_orig, x_fake), dim = 1)
  y_concat = torch.cat(( y_fake_orig, y_fake), dim = 1)
  x_concat_ = x_concat.detach()
  y_concat_ = y_concat.detach()

  x_fake_dis = Discriminator_loss(D_x, x_concat)
  y_fake_dis = Discriminator_loss(D_y, y_concat)

  label_real = Variable(torch.ones(y_fake_dis.size())).to(device)
  adv_loss = MSELoss(x_fake_dis, label_real) + MSELoss(y_fake_dis, label_real)
  adv_loss = adv_loss.to(device)
  #Cycle loss
  Cycle_loss = (L1_loss(re_x, images_x) + L1_loss(re_y, images_y)+L1_loss(re_x_orig, images_orig_x) + L1_loss(re_y_orig, images_orig_y))*lambda_
  #Identity loss
  ID_loss = (L1_loss(x_id, images_x) + L1_loss(x_id_orig, images_orig_x)+ L1_loss(y_id, images_y) + L1_loss(y_id_orig, images_orig_y))*lambda_
  #Context loss
  weight_xy = return_cuda_tensor(get_weights(x_fake,images_y))
  weight_yx = return_cuda_tensor(get_weights(y_fake,images_x))
  Ctx_loss = (L1_loss(weight_yx * images_orig_x, weight_yx* y_fake_orig) + L1_loss(weight_xy*images_orig_y, weight_xy*x_fake_orig))*lambda_
  Total_loss = adv_loss + Cycle_loss + ID_loss + Ctx_loss
  return x_concat_,y_concat_, Total_loss
  
def calc_dis_loss(images_x, sampled_x, images_y, sampled_y):
  x_real_loss = Discriminator_loss(D_x, images_x)
  x_fake_loss = Discriminator_loss(D_x, sampled_x)
  y_real_loss = Discriminator_loss(D_y, images_y)
  y_fake_loss = Discriminator_loss(D_y, sampled_y)
  label_real = Variable(torch.ones(y_fake_loss.size())).to(device)
  label_fake = Variable(torch.zeros(y_fake_loss.size())).to(device)
  err_dx = MSELoss(x_real_loss, label_real) + MSELoss(x_fake_loss,label_fake)
  err_dy = MSELoss(y_real_loss, label_real) + MSELoss(y_fake_loss,label_fake)
  return err_dx, err_dy

def array_of_images(images):
  out= np.ones((batchSize, 9, 4, 70, 70))
  out= torch.tensor(out)
  for index in range(0, images.shape[0]):
    # print(images.shape)
    count = 0
    x=np.ones((3,70,70))
    x = torch.tensor(x)
    for i in range(0, 150,40):
      if(i<=80):
        for j in range(0, 100,70):
          if(j<=30):
            x = images[index, :, i:i+70, j:j+70]
            out[index][count] = x
            count+=1
#   print("count", count)
  return return_cuda_tensor(out)

def Discriminator_loss(D, images):
  arr_x = array_of_images(images)
  dis = 0 
  for index in range(0, arr_x.shape[1]):
    dis += D.forward(arr_x[:, index, :, :, :])
  dis = dis/arr_x.shape[1]
  return(dis)

def remove_transparency(im, bg_colour=(255, 255, 255)):
  if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
    alpha = im.convert('RGBA').split()[-1]
    bg = Image.new("RGBA", im.size, bg_colour + (255,))
    bg.paste(im, mask=alpha)
    return bg
  else:
    return im
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
from models import *
from utils import *
#Cuda 
use_cuda = torch.cuda.is_available()
print('gpu status ===',use_cuda)
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

#Parameters
imageSize = (150,100)
batchSize = 1  
transform_1 = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(),])
transform_2 = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(),]) 
lr_d = 0.0001
lr_g = 0.0002
lambda_ = 10.00


#Initialising the networks
G_xy = ResnetGenerator().to(device)
G_yx = ResnetGenerator().to(device)
D_x = Discriminator().to(device)
D_y = Discriminator().to(device)


#initialising the optimisers
optimiser_g_xy = optim.Adam(G_xy.parameters(), lr= lr_g, betas= (0.5, 0.999))
optimiser_g_yx = optim.Adam(G_yx.parameters(), lr= lr_g, betas= (0.5, 0.999))
optimiser_d_x = optim.Adam(D_x.parameters(),lr= lr_d,  betas= (0.5, 0.999))
optimiser_d_y = optim.Adam(D_y.parameters(),lr= lr_d,  betas= (0.5, 0.999))

#Learning rate schdulers
lr_scheduler_g_xy = torch.optim.lr_scheduler.StepLR(optimiser_g_xy, step_size = 1, gamma = 0.1)
lr_scheduler_g_yx = torch.optim.lr_scheduler.StepLR(optimiser_g_yx, step_size = 1, gamma = 0.1)
lr_scheduler_d_x = torch.optim.lr_scheduler.StepLR(optimiser_d_x, step_size = 1, gamma = 0.1)
lr_scheduler_d_y = torch.optim.lr_scheduler.StepLR(optimiser_d_y, step_size = 1, gamma = 0.1)

#Reading the CCP dataset
X = []
Y = []
orig_X = []
orig_Y = []
data_count = 0
for images in os.listdir(dir_):
  data_count+=1
  #if(data_count<200):
  print("dataset reading going on", data_count)
  if(images.endswith("_skirt.jpg")):
    name = images.split("_skirt.jpg")[0]+".jpg"
    image = Image.open(dir_+name)	
    image_t = transform_2(image)
    # image_t.to(device)
    orig_X.append(image_t)
    image.close()
    image = Image.open(dir_+images)	
    image = remove_transparency(image).convert('L')
    image_t = transform_1(image)
    # image_t.to(device)
    X.append(image_t)
    image.close()
  if(images.endswith("_jeans.jpg") or images.endswith("_pants.jpg")):
    if(images.endswith("_jeans.jpg")):
      name = images.split("_jeans.jpg")[0]+".jpg"
    if(images.endswith("_pants.jpg")):
      name = images.split("_pants.jpg")[0]+".jpg"
    image = Image.open(dir_+name)	
    image_t = transform_2(image)
    # image_t.to(device)
    orig_Y.append(image_t)
    image.close()
    image = Image.open(dir_+images)	
    image = remove_transparency(image).convert('L')
    image_t = transform_1(image)
    # image_t.to(device)
    Y.append(image_t)
    image.close()

# Adding nextra examples using random sampling to make sizes equal
for i in range(0, len(X)):
  X.append(X[i])
  orig_X.append(orig_X[i])
for j in range(0, 125):
  num = np.random.randint(0, len(X))
  X.append(X[num])
  orig_X.append(orig_X[num]) 
print("X", len(X))
print("Y", len(Y))
print("Orig x", len(orig_X))
print("orig Y", len(orig_Y))
dataloader_x = torch.utils.data.DataLoader(X, batch_size = batchSize, shuffle = False, num_workers = 2) 
dataloader_y = torch.utils.data.DataLoader(Y, batch_size= batchSize, shuffle = False, num_workers=2)
dataloader_orig_x = torch.utils.data.DataLoader(orig_X, batch_size = batchSize, shuffle = False, num_workers = 2) 
dataloader_orig_y = torch.utils.data.DataLoader(orig_Y, batch_size= batchSize, shuffle = False, num_workers=2)
print("data loading done")
  

def train(epoch):
  print("*************************************Starting epoch********************************************* ", epoch+1)
  samples_x=[]  
  samples_y=[]
 
  for (n, images_x), (n1, images_orig_x), (n2, images_y), (n3, images_orig_y) in zip(enumerate(dataloader_x), enumerate(dataloader_orig_x), enumerate(dataloader_y), enumerate(dataloader_orig_y)):
    print("----------------------------Image number----------------------------------", n+1)
    images_x = images_x.to(device)
    images_orig_x = images_orig_x.to(device)
    images_y = images_y.to(device)
    images_orig_y = images_orig_y.to(device)
    real_x = torch.cat(( images_orig_x,images_x), dim =1)
    real_y = torch.cat(( images_orig_y, images_y), dim =1)
    x_concat_,y_concat_, Total_loss = calc_gen_loss(images_x, images_orig_x, images_y, images_orig_y)
    print("generator loss",Total_loss.item())
    Total_loss.backward()
    optimiser_g_xy.step()
    optimiser_g_yx.step()
    samples_x.append(x_concat_)
    samples_y.append(y_concat_)
    sample_x = Sample_images(samples_x)
    sample_y = Sample_images(samples_y)
    D_x.zero_grad()
    D_y.zero_grad() 
    err_dx, err_dy = calc_dis_loss(real_x, sample_x, real_y, sample_y)
    err_d = err_dx+err_dy
    print("discriminator loss", err_d.item())
    err_d.backward()
    optimiser_d_x.step()
    optimiser_d_y.step()
    y_dash, b_dash = G_xy.forward(images_x, images_orig_x)
    vutils.save_image(images_orig_x, dir_orig_y+str(epoch+1)+"_"+str(n)+"_"+"real"+".png", normalize = False)
    vutils.save_image(y_dash, dir_orig_y+str(epoch+1)+"_"+str(n)+"_"+"mask"+".png", normalize = False)
    vutils.save_image(b_dash, dir_orig_y+str(epoch+1)+"_"+str(n)+"_"+"orig"+".png", normalize = False)
    x_dash, a_dash = G_yx.forward(images_y, images_orig_y)
    vutils.save_image(images_orig_y, dir_orig_x+str(epoch+1)+"_"+str(n)+"_"+"real"+".png", normalize = False)
    vutils.save_image(x_dash, dir_orig_x+str(epoch+1)+"_"+str(n)+"_"+"mask"+".png", normalize = False)
    vutils.save_image(a_dash, dir_orig_x+str(epoch+1)+"_"+str(n)+"_"+"orig"+".png", normalize = False)
  checkpoint_1 = {'model': Discriminator(),'state_dict': D_x.state_dict(), 'optimizer' : optimiser_d_x.state_dict()}
  checkpoint_2 = {'model': Discriminator(),'state_dict': D_y.state_dict(), 'optimizer' : optimiser_d_y.state_dict()}
  checkpoint_3 = {'model': ResnetGenerator(),'state_dict': G_xy.state_dict(), 'optimizer' : optimiser_g_xy.state_dict()}
  checkpoint_4 = {'model': ResnetGenerator(),'state_dict': G_yx.state_dict(), 'optimizer' : optimiser_g_yx.state_dict()}
  torch.save(checkpoint_1, '/content/drive/My Drive/INSTAGAN/checkpoint_1.pth')
  torch.save(checkpoint_2, '/content/drive/My Drive/INSTAGAN/checkpoint_2.pth')
  torch.save(checkpoint_3, '/content/drive/My Drive/INSTAGAN/checkpoint_3.pth')
  torch.save(checkpoint_4, '/content/drive/My Drive/INSTAGAN/checkpoint_4.pth')
 

#training start   
for epoch in range(0, 200):
  print("---------------------------------------Starting epoch---------------------------------", epoch +1)
  train(epoch)
  if(epoch>=100):
    lr_scheduler_g_xy.step()
    lr_scheduler_g_yx.step()
    lr_scheduler_d_x.step()
    lr_scheduler_d_y.step()
    
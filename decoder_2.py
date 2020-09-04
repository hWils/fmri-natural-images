# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:00:52 2020

@author: holly
"""
import numpy as np
import torch.nn as nn

# decoding model
class decoder(nn.Module):
  def __init__(self):
    super().__init__()
 # def __init__(self):
    self.d0 = nn.Sequential(
        # nn.Linear(in_features = 1024, out_features =1024),
         nn.ConvTranspose2d(in_channels = 1024, out_channels =1024,kernel_size = (3,3),stride =2,padding =1,output_padding =1),
         nn.ELU())
    self.da = nn.Sequential(
        # nn.Linear(in_features = 1024, out_features =1024),
         nn.ConvTranspose2d(in_channels = 1024, out_channels =700,kernel_size = (3,3),stride =2,padding =1,output_padding =1),
         nn.ELU())
    self.d1 = nn.Sequential(
         nn.ConvTranspose2d(in_channels = 700, out_channels =512,kernel_size = (3,3),stride =2,padding =1,output_padding =1),
         nn.ELU())
    self.d2 = nn.Sequential(
         nn.ConvTranspose2d(in_channels = 512, out_channels =384,kernel_size = (3,3),stride =2,padding =1,output_padding =1),
         nn.ELU())
    self.d3 = nn.Sequential(
         nn.ConvTranspose2d(in_channels = 384, out_channels =256,kernel_size = (3,3),stride =2,padding =1,output_padding =1),
         nn.ELU())
    self.d4 = nn.Sequential(
         nn.ConvTranspose2d(in_channels = 256, out_channels =192,kernel_size = (3,3),stride =2,padding =1,output_padding =1),
         nn.ELU())
    self.d5 = nn.Sequential(
         nn.ConvTranspose2d(in_channels = 192, out_channels =1,kernel_size = (3,3),stride =2,padding =1,output_padding =1),   ### 3 -> 1
         nn.ELU())
  def forwardDecoder(self,input, batchSize):
    #  print("input befoe layers :", input.shape)
      input = input.view(batchSize, 1024, 1, 1) ##
      input = self.d0(input)
     # print("d0 1", input.shape)
#      z = input.clone().reshape(batchSize, 1024, 1, 1) ##
     # print("d0_2, 4, 1024 :", input.shape)
      input = self.da(input)
      input = self.d1(input)
     # print("d1, 8 512 : ",input.shape)
      input = self.d2(input)
     # print("d2, 16, 384 : ",input.shape)
      input = self.d3(input)
     # print("d3,32, 256 : ",input.shape)
      input = self.d4(input)
     # print("d4, 64, 192: ",input.shape)
      input = self.d5(input)
     # print("d5 128, 3: ",input.shape)
      return input
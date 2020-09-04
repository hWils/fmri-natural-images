# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:01:09 2020

@author: holly
"""
import numpy as np
import torch.nn as nn


class discriminator(nn.Module):
  def __init__(self):
    super().__init__()
 # def __init__(self):
    self.d0 = nn.Sequential(
         nn.Conv2d(in_channels = 1, out_channels =64,kernel_size = (4,4),stride =2,padding =1),   # 3 -> 1
         nn.ELU())
    self.d1 = nn.Sequential(
         nn.Conv2d(in_channels = 64, out_channels =64,kernel_size = (4,4),stride =2,padding =1),
         nn.ELU())
    self.d2 = nn.Sequential(
         nn.Conv2d(in_channels = 64, out_channels =64,kernel_size = (4,4),stride =2,padding =1),
         nn.ELU())
    self.d3 = nn.Sequential(
         nn.Conv2d(in_channels = 64, out_channels =64,kernel_size = (4,4),stride =2,padding =1),
         nn.ELU())
    self.d4 = nn.Sequential(
         nn.Conv2d(in_channels = 64, out_channels =64,kernel_size = (4,4),stride =2,padding =1),
         nn.ELU())
    self.d5 = nn.Sequential(
        nn.Linear(in_features = 64*16, out_features =1),
      #  nn.Conv2d(in_channels = 64, out_channels =1,kernel_size = (4,4),stride =3,padding =1),
        nn.Sigmoid())
    
  def forwardDiscriminator(self,input):
     # print("Discriminator")
      input = self.d0(input)
    #  print(input.shape)
      input = self.d1(input)
    #  print(input.shape)
      input = self.d2(input)
    #  print(input.shape)
      input = self.d3(input)
      intermediate = input
     # print(input.shape)
      input = self.d4(input)
    #  print(input.shape)
    #  print('unflat shape : ',input.shape)
      input = input.view(50,64*16)
     # print("a :,", input)
      #input = input.view(1,64)
     # print("b :, input)
      input = self.d5(input)
    #  print("End of disc: ", input)
     # print(input.shape)
      return input, intermediate
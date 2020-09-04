# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:00:00 2020

@author: holly
"""


import numpy as np
import torch.nn as nn


class encoder(nn.Module):
 # def __init__(self):
  def __init__(self):
    super().__init__()

    self.c1 = nn.Sequential(
         nn.Conv2d(in_channels = 1, out_channels =192,kernel_size = (3,3),stride =2,padding =1),   # 3 -> 1
         nn.ELU())
    self.c2 = nn.Sequential(
         nn.Conv2d(in_channels = 192, out_channels =256,kernel_size = (3,3),stride =2,padding =1),
         nn.ELU())
    self.c3 = nn.Sequential(
         nn.Conv2d(in_channels = 256, out_channels =384,kernel_size = (3,3),stride =2,padding =1),
         nn.ELU())
    self.c4 = nn.Sequential(
         nn.Conv2d(in_channels = 384, out_channels =512,kernel_size = (3,3),stride =2,padding =1),
         nn.ELU())
    self.c5 = nn.Sequential(
         nn.Conv2d(in_channels = 512, out_channels =768,kernel_size = (3,3),stride =2,padding =1),
         nn.ELU())
    self.fc1_mu = nn.Sequential(
         nn.Linear(in_features = 768*16, out_features =1024),
         nn.ELU())
    self.fc1_logvar = nn.Sequential(
         nn.Linear(in_features = 768*16, out_features =1024),
         nn.ELU())
  
  def forwardEncoder(self,input, batchsize):
    input = self.c1(input)
   # print(input.shape)
    input = self.c2(input)
   # print(input.shape)
    input = self.c3(input)
  #  print(input.shape)
    input = self.c4(input)
 #   print(input.shape)
    input = self.c5(input)
 #   print("before flattening: ", input.shape)
   # input = input.view()
   # print("hello", input.shape)
  #  input = input.view(-1,768) #flattening
    input = input.view(batchsize,768*16) #flattening
  #  print("bye", input.shape)
 #   print("flattened : ",out.shape)
    mu = self.fc1_mu(input)
    logvar =self.fc1_logvar(input)
   # print(input.shape)
    return mu, logvar
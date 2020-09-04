# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 19:06:21 2020

@author: holly
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from torchsummary import summary
import encoder
import decoder
import discriminator


read_dictionary = np.load('C:\\Users\\holly\\Downloads\\my_file.npy',allow_pickle='TRUE').item()
dat = read_dictionary

# PROCESS IMAGES
# training a VAE on the image dataset
images = dat['stimuli']
tensor_image = torch.Tensor(images)
tensor_image= torch.unsqueeze(tensor_image, axis = 1)
#images = np.swapaxes(images,2,3)  1
print(tensor_image.shape)

"""
images = np.repeat(images[..., np.newaxis], 3, -1)
images = np.swapaxes(images,1,3)
images = np.swapaxes(images,2,3)
print(images.shape)
"""



# During training random sample from the learned ZDIMS-dimensional; normal distribution; during inference its mean.
def sample(mu,logvar):
  std = torch.exp(0.5*logvar)
  eps = torch.randn_like(std) # this is where the random sampling is occuring - random seed?
  z = std*eps + mu
  return z, logvar, mu


def kl_loss(mu, logvar):
  kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return kl_loss



# three networks with separate optimisers
practice_encoder = encoder.encoder()
practice_decoder = decoder.decoder()
practice_discrim = discriminator.discriminator()
optimizerDiscrim = optim.Adam(practice_discrim.parameters(),lr=0.0001)
optimizerEncod = optim.Adam(practice_encoder.parameters(),lr=0.0001)
optimizerGen = optim.Adam(practice_decoder.parameters(),lr=0.0001)




# batches and epochs
"""
one epoch = one forward pass and one backward pass of all the training examples
batch size = the number of training examples in one forward/backward pass. 
            The higher the batch size, the more memory space you'll need.


batchSize = 50, epochs = 12

"""



batch_size = 50
#tensor_image = torch.Tensor(images) 1
print(tensor_image.shape[0])
numImages = tensor_image.shape[0]

#for img in tensor_image:
for epoch in range(50):
    print("Beginning new epoch :", epoch)
   # for batch in range(0,numImages-batch_size+1, batch_size): 
    for batch in range(0,150+1, batch_size): 
        
        
      print("image indexed from: ", batch)
      #img = img.unsqueeze(dim = 0)
    #  print(tensor_image[:batch_size].shape, img.shape)
      image_batch = tensor_image[batch:batch+batch_size]
     # mu, logvar = practice_encoder.forwardEncoder(tensor_image[:batch_size])
     
     # forward for encoder
      mu, logvar = practice_encoder.forwardEncoder(image_batch,batch_size)
      # sampling to get means/stds from encoderOutput
     # print("mu :", mu.shape)
      z, logvar, mu = sample(mu, logvar) #actual
     # print("latent variables z : ", z.shape)
      #z_t, logvar_t, mu_t = sample(0,1) # target
    
    
      # forward for generator
      decoderOutput = practice_decoder.forwardDecoder(z, batch_size)
     # print("BEFORE: ", practice_encoder.c1[0].weight)
      
  #    print("BEFORE: ", practice_decoder.d0[0].weight)
      
    # UPDATES
      target_real = torch.ones(batch_size)
      target_fake = torch.zeros(batch_size)
      lossBCED = nn.BCELoss()
      lossBCEG = nn.BCELoss()
     # print("THE SHAPES: ", image_batch.shape, decoderOutput.shape)
      real = practice_discrim.forwardDiscriminator(image_batch)
     # print("real complete")
      fake = practice_discrim.forwardDiscriminator(decoderOutput)
     # print("fake complete")
     # #discriminator loss and zero grads
      lossD_real = lossBCED(real,target_real)
    #  print(lossD_real)
      lossD_fake = lossBCED(fake,target_fake)
      #lossD = lossD_real + lossD_fake
      print("accuracy real - good is 50", np.sum(real.detach().numpy()), np.sum(target_real.detach().numpy()))
      print("accuracy fake - good is 0 ", np.sum(fake.detach().numpy()), np.sum(target_fake.detach().numpy()))
      # guessing fakes as being real incorrectly.
      
      
    #  print("loss discriminator, ", lossD)
      optimizerDiscrim.zero_grad()
    
      # generator loss and zerograds
      target_fake = torch.ones(batch_size)
      lossG = lossBCEG(fake, target_fake)
      print("loss generator, ", lossG)
      optimizerGen.zero_grad()
    
      # encoder loss and zero grads
      kl = kl_loss(mu,logvar)
      optimizerEncod.zero_grad()
    
      # back propogation all.
      lossD_real.backward(retain_graph=True)
      lossD_fake.backward(retain_graph=True)
     # lossD.backward(retain_graph=True)
      lossG.backward(retain_graph=True)  # maybe the problem is here
      kl.backward(retain_graph=True)
      optimizerDiscrim.step()
      optimizerGen.step()
      optimizerEncod.step()
      
      baby = practice_decoder.forwardDecoder(mu[15], 1)
      baby = baby.detach().numpy()
      baby = np.squeeze(baby)
      baby = np.squeeze(baby)
      print(baby.shape)
      plt.imshow(baby, cmap = 'gray')
      plt.show()

      
    #  print("AFTER: ", practice_encoder.c1[0].weight)
      
      
   
    
    
trainImage = tensor_image[0:50]

mu, logvar = practice_encoder.forwardEncoder(trainImage, batch_size)
      # sampling to get means/stds from encoderOutput
     # print("mu :", mu.shape)
z, logvar, mu = sample(mu, logvar) #actual

print(mu)
print(z)
print("logvar : ", logvar)
     # print("latent variables z : ", z.shape)
      #z_t, logvar_t, mu_t = sample(0,1) # target
    
    
      # forward for generator
      
# means for two images
print("twentyy mean :, ", mu[20][0:20])
print("ten mean:, ", mu[10][0:20])



decoderOutput = practice_decoder.forwardDecoder(mu, batch_size)
twenty = decoderOutput[20].detach().numpy()
ten = decoderOutput[10].detach().numpy()
print("twentyy decoder output:, ", twenty[0:20])
print("ten decoder output:, ", ten[0:20])
#cat = np.squeeze(cat)
#cat = np.swapaxes(cat,0,2)
plt.imshow(cat, cmap = 'gray')
dog = np.swapaxes(trainImage[49],0,2)
dog = np.swapaxes(dog, 0,1)
plt.imshow(dog)



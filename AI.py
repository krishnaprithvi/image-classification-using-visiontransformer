#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install torch


# In[2]:


pip install torchvision


# In[3]:


import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
from helper_functions import set_seeds


# In[4]:


device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[5]:


# 1. Get pretrained weights for ViT-Base
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 

# 2. Setup a ViT model instance with pretrained weights
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# 3. Freeze the base parameters
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False
    
# 4. Change the classifier head 
class_names = ['cataract','diabetic_retinopathy','glaucoma','normal']

set_seeds()
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
# pretrained_vit # uncomment for model output 


# In[6]:


pip install torchinfo


# In[7]:


from torchinfo import summary

# Print a summary using torchinfo (uncomment for actual output)
summary(model=pretrained_vit, 
        input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)


# #### Notice how only the output layer is trainable, where as, all of the rest of the layers are untrainable (frozen).

# In[8]:


# Setup directory paths to train and test images
train_dir = '/Users/krishnaprithvibattula/Documents/AI/ImageClassificationUsingVisionTransformer/Training'
test_dir = '/Users/krishnaprithvibattula/Documents/AI/ImageClassificationUsingVisionTransformer/Testing'


# Remember, if you're going to use a pretrained model, it's generally important to ensure your own custom data is transformed/formatted in the same way the data the original model was trained on.

# In[9]:


# Get automatic transforms from pretrained ViT weights
pretrained_vit_transforms = pretrained_vit_weights.transforms()
print(pretrained_vit_transforms)


# ## And now we've got transforms ready, we can turn our images into DataLoaders using the create_dataloaders()

# In[10]:


import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):

  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=0,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=0,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names


# In[11]:


# Setup dataloaders
train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(train_dir=train_dir,
                                                                                                     test_dir=test_dir,
                                                                                                     transform=pretrained_vit_transforms,
                                                                                                     batch_size=32) # Could increase if we had more samples, such as here: https://arxiv.org/abs/2205.01580 (there are other improvements there too...)


# In[12]:


from goingmodular.going_modular import engine

# Create optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), 
                             lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the classifier head of the pretrained ViT feature extractor model
set_seeds()
pretrained_vit_results = engine.train(model=pretrained_vit,
                                      train_dataloader=train_dataloader_pretrained,
                                      test_dataloader=test_dataloader_pretrained,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=15,
                                      device=device)


# ## That's the power of transfer learning!
# 
# We managed to get outstanding results with the same model architecture, except our custom implementation was trained from scratch (worse performance) and this feature extractor model has the power of pretrained weights from ImageNet behind it.

# # Let's make Prediction:

# In[26]:


# Import function to make predictions on images and plot them 
from goingmodular.going_modular.predictions import pred_and_plot_image

# Setup custom image path
custom_image_path = ["/Users/krishnaprithvibattula/Documents/AI/ImageClassificationUsingVisionTransformer/Testing/glaucoma/_8_5210275.jpg", "/Users/krishnaprithvibattula/Documents/AI/ImageClassificationUsingVisionTransformer/Testing/glaucoma/_180_6449017.jpg", "/Users/krishnaprithvibattula/Documents/AI/ImageClassificationUsingVisionTransformer/Testing/glaucoma/_8_5210275.jpg", "/Users/krishnaprithvibattula/Documents/AI/ImageClassificationUsingVisionTransformer/Testing/glaucoma/_269_7481383.jpg"]

# Predict on custom image
for image in custom_image_path:
    pred_and_plot_image(model=pretrained_vit,
                    image_path=image,
                    class_names=class_names)


# In[ ]:





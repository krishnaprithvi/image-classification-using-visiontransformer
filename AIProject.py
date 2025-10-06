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
import torchvision.models as models

from torch import nn
from torchvision import transforms
from helper_functions import set_seeds


# In[4]:


device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[5]:


# Obtain pretrained weights for ViT-Base model
pretrained_weights = models.ViT_B_16_Weights.DEFAULT 

# Instantiate a ViT model with the pretrained weights
pretrained = models.vit_b_16(weights=pretrained_weights).to(device)

# Freeze the parameters of the base model to prevent them from being updated during training
for parameter in pretrained.parameters():
    parameter.requires_grad = False
    
# Modify the classifier head for the specific classification task
class_names = ['cataract','diabetic_retinopathy','glaucoma','normal']

# Set random seeds for reproducibility
set_seeds()

# Replace the classifier head with a new linear layer
pretrained.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)


# In[6]:


pip install torchinfo


# In[7]:


from torchinfo import summary

# Generate a summary using torchinfo (uncomment to display actual output)
summary(model=pretrained, 
        input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)


# #### It's worth noting that only the output layer is set to be trainable, while all other layers are frozen and untrainable.

# In[8]:


# Define directory paths for training and testing images
training_dir = '/Users/krishnaprithvibattula/Documents/AI/ImageClassificationUsingVisionTransformer/Training'
testing_dir = '/Users/krishnaprithvibattula/Documents/AI/ImageClassificationUsingVisionTransformer/Testing'


# Remember, if you're going to use a pretrained model, it's generally important to ensure your own custom data is transformed/formatted in the same way the data the original model was trained on.

# In[9]:


# Obtain automatic transforms from pretrained ViT weights
pretrained_vit_transforms = pretrained_weights.transforms()
print(pretrained_vit_transforms)


# ## With our transformations prepared, we can now convert our images into DataLoaders using the create_dataloaders() function.

# In[10]:


import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    training_dir: str, 
    testing_dir: str, 
    transforms: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):

  # Create dataset(s) from ImageFolder for training and testing
  train_data = datasets.ImageFolder(training_dir, transform=transforms)
  test_data = datasets.ImageFolder(testing_dir, transform=transforms)

  # Extract class names from the training dataset
  class_names = train_data.classes

  # Create DataLoader objects for training and testing dataset(s)
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


# Setting up dataloaders
train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(training_dir=training_dir,
                                                                                                     testing_dir=testing_dir,
                                                                                                     transforms=pretrained_vit_transforms,
                                                                                                     batch_size=32) # Could increase if we had more samples, such as here: https://arxiv.org/abs/2205.01580 (there are other improvements there too...)


# In[12]:


from goingmodular.going_modular import engine

# Define the optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained.parameters(), 
                             lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the classification head of the pre-trained ViT feature extraction model
set_seeds() # Set random seeds for reproducibility
pretrained_vit_results = engine.train(model=pretrained,
                                      train_dataloader=train_dataloader_pretrained,
                                      test_dataloader=test_dataloader_pretrained,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=15,
                                      device=device)


# ## The effectiveness of transfer learning shines through!
# 
# We've achieved remarkable results using the same model architecture. However, our custom implementation, trained from scratch, yielded inferior performance compared to this feature extractor model, which benefits from pre-trained weights from ImageNet.

# # Now, let's proceed with making predictions:

# In[14]:


# Import function for image prediction and plotting from the modular module
from goingmodular.going_modular.predictions import pred_and_plot_image

# Define custom image paths for prediction
custom_image_path = ["/Users/krishnaprithvibattula/Documents/AI/ImageClassificationUsingVisionTransformer/Testing/glaucoma/_8_5210275.jpg",
                     "/Users/krishnaprithvibattula/Documents/AI/ImageClassificationUsingVisionTransformer/Testing/glaucoma/_180_6449017.jpg",
                     "/Users/krishnaprithvibattula/Documents/AI/ImageClassificationUsingVisionTransformer/Testing/glaucoma/_8_5210275.jpg",
                     "/Users/krishnaprithvibattula/Documents/AI/ImageClassificationUsingVisionTransformer/Testing/glaucoma/_269_7481383.jpg"]

# Predict and plot each image
for image in custom_image_path:
    pred_and_plot_image(model=pretrained,
                    image_path=image,
                    class_names=class_names)


# In[ ]:





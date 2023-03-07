
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm, trange
import torchvision.transforms as transforms
import torch.nn as nn 

from Loss import Dice_CE_Loss
from one_hot_encode import one_hot,one_hot1
from data_loader import loader
from Model import build_unet
from PIL import Image
from torch.utils.data import Dataset
import os

def Dice_Loss(input,target):
    smooth=1
    intersection = (input * target).sum()
    dice_loss    = 1- (2.*intersection + smooth )/(input.sum() + target.sum() + smooth)
    return dice_loss

def BCE_loss(input,target):
    sigmoid_f      = nn.Sigmoid()
    input= sigmoid_f(input)
    B_cross_entropy = nn.BCELoss(reduction='mean')
    return B_cross_entropy(input, target) 

def Dice_BCE_Loss(input,target):
    return Dice_Loss(input,target) + BCE_loss(input,target)


def softmax_manuel(input):
    return (torch.exp(input) / torch.sum(torch.exp(input)))

def CE_loss_manuel(input,target):
    return torch.mean(-torch.sum(torch.log(softmax_manuel(input)) * (target)))

def CE_loss(inputs,target):
    input_F=torch.flatten(input=inputs,start_dim=0,end_dim=2)
    target_F=torch.flatten(input=target,start_dim=0,end_dim=2)
    #inp=inputs.flatten()
    #tar=target.flatten()
    cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    return cross_entropy(input_F,target_F)

def main():
    #n_classes   = 2
    batch_size  = 2
    num_workers = 2
    epochs      = 30
    l_r         = 0.001

    train_loader,test_loader = loader(batch_size,num_workers,shuffle=True)
    
    model     = build_unet()
    optimizer = Adam(model.parameters(), lr=l_r)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    for epoch in trange(epochs, desc="Training"):

        loss = 0.0
        i=0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):

            images,labels   = batch        
            model_output    = model(images)
            model_output    = torch.transpose(model_output,1,3)
            
            target          = one_hot(labels,2)
            target1         = one_hot1(labels,1)

            ce_loss_m       = CE_loss_manuel(model_output, target)
            ce_loss1        = CE_loss(model_output, target1)

            loss     += loss.detach().item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} loss: {loss}")

        im_test    = np.array(images[0]*255,dtype=int)
        im_test    = np.transpose(im_test, (2, 1, 0))
        label_test = np.array(labels[0],dtype=int)
        label_test = np.transpose(label_test)

        #prediction = torch.argmax(prediction,dim=2)
        prediction  = model_output[0].squeeze()
        prediction  = prediction.detach().numpy()

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(im_test)
        plt.subplot(1, 3, 2)
        plt.imshow(label_test)
        plt.subplot(1, 3, 3)
        plt.imshow(prediction*255)


'''
    model     = UNET((1, 28, 28),n_heads=2, output_dim=10,mlp_layer_size=8)    
    optimizer = Adam(model.parameters(), lr=l_r)
    criterion = Dice_BCE_Loss()
    
    for epoch in trange(epochs, desc="Training"):
        train_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            
            image, masks    = batch
            predicted_masks  = model(x)
            loss = criterion(predicted_masks, masks)
            train_loss += loss.detach().cpu().item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss}")

        


    train_im_path   = "train/images"
    train_mask_path = "train/masks"
    test_im_path    = "test/images"
    test_mask_path  = "test/masks"

    train_im_dir_list   = os.listdir(train_im_path) 
    train_mask_dir_list = os.listdir(train_mask_path) 
    test_im_dir_list    = os.listdir(train_mask_path) 
    test_mask_dir_list  = os.listdir(train_im_path) 

    for idx,_ in enumerate(train_im_dir_list):
        image_dir = os.path.join(train_im_path,train_im_dir_list[idx])
        image = Image.open(image_dir)


'''


if __name__ == "__main__":
   main()
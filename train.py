
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm, trange
import torchvision.transforms as transforms
import torch.nn as nn 

from Loss import Dice_CE_Loss
from one_hot_encode import one_hot
from data_loader import loader
from Model import build_unet
from PIL import Image
from torch.utils.data import Dataset
import os

def Dice_Loss(input,target):
        smooth=1
        input =torch.flatten(input=input)
        target=torch.flatten(target)
        intersection = (input * target).sum()
        dice_loss    = 1- (2.*intersection + smooth )/(input.sum() + target.sum() + smooth)
        return dice_loss
 
def CE_loss(input,target):
        input=torch.flatten(input=input)
        target=torch.flatten(target)
        sigmoid_f      = nn.Sigmoid()
        input= sigmoid_f(input)
        cross_entropy = nn.BCELoss(reduction='mean')
        return cross_entropy(input, target) 
    
def Dice_BCE_Loss(input,target):
    return Dice_Loss(input,target) + CE_loss(input,target)


def main():
    #n_classes   = 2
    batch_size  = 2
    num_workers = 2
    epochs      = 10
    l_r         = 0.05

    train_loader,test_loader = loader(batch_size,num_workers,shuffle=True)
    
    model = build_unet()
    optimizer = Adam(model.parameters(), lr=l_r)
    
    for epoch in trange(epochs, desc="Training"):

        train_loss = 0.0
        i=0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):

            images,labels   = batch        
            model_output    = model(images)
            #targets         = one_hot(labels,n_classes)
            inputs          = torch.transpose(model_output,3,2)
            inputs=inputs.to(dtype=float)
            model_output=model_output.to(dtype=float)

            train_loss      = Dice_BCE_Loss(model_output,labels)
            #CE_lossmo      = loss.CE_loss()
            #CE_loss_manuel = loss.CE_loss_manuel()
            #dice_loss      = loss.Dice_Loss()
            #train_loss      = loss.Dice_CE_loss()
            train_loss     += train_loss.detach().item() / len(train_loader)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            i=i+1 
            if i%2==0:
                im_test    = np.array(images[0]*255,dtype=int)
                im_test    = np.transpose(im_test, (2, 1, 0))
                label_test = np.array(labels[0],dtype=int)
                label_test = np.transpose(label_test)
                #prediction = inputs[0]
                #prediction = torch.argmax(prediction,dim=2)
                prediction  = inputs[0].squeeze()
                prediction  = prediction.detach().numpy()

                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(im_test)
                plt.subplot(1, 3, 2)
                plt.imshow(label_test)
                plt.subplot(1, 3, 3)
                plt.imshow(prediction*255)

            print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss}")


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
import numpy as np
import os
from tqdm import tqdm
import cv2
import torch.nn.functional as F
import torch as t
import torchio as tio
import monai
from torch.utils import data
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
mask_names = ['Lung_L','Lung_R','Heart','SpinalCord','Esophagus']

def load_image(image):
    image_tensor = image.reshape(1,image.shape[0],image.shape[1],image.shape[2]).astype(np.float32)
    return t.from_numpy(image_tensor[None,...])


def post_process(pred):
    #NCDHW input
    classes = pred.shape[1]    
    #pred = F.softmax(pred,dim=1)
    maxvar = t.zeros_like(pred)
    for i in range(classes):
        maxvar[:,i,...] = t.argmax(pred,1)==i
    del pred
    transform = monai.transforms.KeepLargestConnectedComponent([1]) #tio.transforms.KeepLargestComponent()
    
    newmask = t.zeros_like(maxvar)
    newmask[0,0,...] = maxvar[0,0,...]
    for i in range(1,maxvar.shape[1]):
        newmask[0,i,...] = transform(maxvar[:,i,...].permute(0,3,2,1)).permute(0,3,2,1)
    #print(newmask.shape)
    
    return newmask
        
def save_mask(masks,name):

    for i in range(masks.shape[0]):
        try:    
            os.makedirs(save_path+name+"/masks")
            np.save(save_path+name+"/masks/"+mask_names[i-1]+".npy",masks[i])
        except FileExistsError:
            np.save(save_path+name+"/masks/"+mask_names[i-1]+".npy",masks[i])
    print("Saved mask files for "+name)
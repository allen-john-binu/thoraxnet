#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##Cropping
import numpy as np
import os
from tqdm import tqdm
import cv2
import time
import torch.nn.functional as F
import torch as t
from crop_model import UnetModel as UnetModelCrop
from scipy.ndimage import zoom
from torch.utils import data

mask_names = ['Lung_L','Lung_R','Heart','SpinalCord','Esophagus']
weights = {
    'Lung_L':"/workstation/seenia/autocontouring/Thoracic_Autocontouring/manu/3D-Model/BoundingBox/weights/LungL_Box3D_2.997848853468895",
     'Lung_R':"/workstation/seenia/autocontouring/Thoracic_Autocontouring/manu/3D-Model/BoundingBox/weights/LungR_Box3D_2.4519119895994663",
     'Heart':"/workstation/seenia/autocontouring/Thoracic_Autocontouring/manu/3D-Model/BoundingBox/weights/Heart_Box3D_5.692844539880753",
     'SpinalCord':"/workstation/seenia/autocontouring/Thoracic_Autocontouring/manu/3D-Model/BoundingBox/weights/SpinalCord_Box3D_4.82074498385191",
     'Esophagus':"/workstation/seenia/autocontouring/Thoracic_Autocontouring/manu/3D-Model/BoundingBox/weights/Esophagus_Box3D_9.024686396121979"
        }
def load_image(image):
    image_tensor = image.reshape(1,image.shape[0],image.shape[1],image.shape[2]).astype(np.float32)
    return t.from_numpy(image_tensor[None,...])


# In[ ]:


def calc_crop(image):
    x_test = load_image(image)
    crops = {}
    for mask in mask_names:
        model = UnetModelCrop(in_channels=1,out_classes = 2)
        model.load_state_dict(t.load(weights[mask],map_location='cuda:0')) #Loading model weights for current organ
        model.eval()
        with t.no_grad():
            
            x_test = x_test.to('cuda:0')
            o = model(x_test)
            
            o1 = F.softmax(o,dim=1)
            
            o_test=o1.cpu()
            o_test=np.round(np.array(o_test),0)
            o_test=o_test.astype(np.uint8)
            data=o_test[0][1]
            #starttime = time.time()
            #data = zoom(data, (4, 4,4))
            #print(time.time()-starttime)
            minxlst, minylst, minzlst, maxxlst, maxylst, maxzlst = [], [], [], [], [], []
            zlst=[]
            
            for sl in range(data.shape[0]):
                if np.sum(data[sl])>0 :
                    zlst.append(sl)
            minzlst.append(min(zlst))
            maxzlst.append(max(zlst))
            xmin,xmax,ymin,ymax =[],[],[],[]
            for sl in zlst:
                result = np.where(data[sl] == 1)
                xmin.append(np.min(result[0]))
                xmax.append(np.max(result[0]))
                ymin.append(np.min(result[1]))
                ymax.append(np.max(result[1]))
            minxlst.append(np.min(xmin))
            minylst.append(np.min(ymin))
            maxxlst.append(np.max(xmax))
            maxylst.append(np.max(ymax))
            #print(nam, x_test.shape, data.shape,"( ", minzlst, "," , maxzlst, ",", minxlst, ",", maxxlst, ",", minylst, ",", maxylst, ")")
        
        crops[mask] = {"minz":minzlst[0]*4,"maxz":maxzlst[0]*4,"minx":minxlst[0]*4,"maxx":maxxlst[0]*4,"miny":minylst[0]*4,"maxy":maxylst[0]*4}
        del model
    #print(crops)
    print("Cropping Completed")
    return crops


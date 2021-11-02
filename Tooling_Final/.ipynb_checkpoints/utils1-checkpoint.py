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

def get_images_masks(patient,organ):
    data = np.load(patient+"/img_crp_v2.npy").astype(np.float16)
    masks = []
    
    for name in mask_names[2:3]:
        if organ is not None:
            mask = np.load(patient+"/structure/"+organ+"_crp_v2.npy").astype(np.uint8)
        else:
            mask = np.load(patient+"/structure/SpinalCord_crp_v2.npy").astype(np.uint8)
        masks.append(mask)
    #assert len(masks) == 5
    return data,masks

def import_data(data_path,istest,isvalidate,organ):
    ct_patients = [os.path.join(data_path,name) for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
    ct_data = []
    ct_patients.sort(key = lambda x : x[-3:])

    if istest:
        ct_patients = ct_patients[48:60]
    elif isvalidate:
        ct_patients = ct_patients[36:48]
    else:
        ct_patients = ct_patients[:36]
    for patient in tqdm(ct_patients):
        scan = {}
        scan["image"],scan["mask"] = get_images_masks(patient,organ)
        scan["name"] = patient[len(data_path)+1:]
        ct_data.append(scan)
    return ct_data

class CT_Dataset(data.Dataset):
    def __init__(self,path,istest = False,isvalidate=False,istransform = True, organ=None):
        self.data = import_data(path,istest,isvalidate,organ)
        self.transform = istransform
        self.istest = istest
        self.isvalidate = isvalidate
        
    def __getitem__(self,i):
        img = self.data[i]["image"]
        name = self.data[i]["name"]
        maskindex = [0]
        masks = [mask.astype(np.uint8).reshape((1,img.shape[0],img.shape[1],img.shape[2])) for mask in [self.data[i]["mask"][j] for j in maskindex]]
        img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
        if(self.transform):
            img_t, masks_t = self.transform3DCT( t.from_numpy((np.concatenate([img.astype(np.float32)] + masks,axis=0)).astype(np.float32)))
            masks_t[masks_t<0.8]=0
            masks_t[masks_t>=0.8]=1
        else:
            img_t = img[0,...].astype(np.float32)
            masks_t = t.from_numpy(np.concatenate(masks, axis=0).astype(np.float32))
        non_mask = np.zeros_like(masks_t[0,...]).astype(np.uint8)
        masks_new = []
        for i in range(masks_t.shape[0]):
            non_mask = np.logical_or(non_mask, masks_t[i,...].numpy()).astype(np.uint8)
            masks_new.append(masks_t[i,...].reshape((1,)+img.shape[1:]))
        non_mask = (1 - non_mask).reshape((1,)+non_mask.shape)
        if self.istest:
            return img_t[None,...], t.from_numpy(np.concatenate([non_mask]+masks_new, axis=0)),name
        else:
            return img_t[None,...], t.from_numpy(np.concatenate([non_mask]+masks_new, axis=0)),name
        
              
    def transform3DCT(self,image):
                
        spatial_transforms = {
            tio.RandomMotion(degrees=4,translation=7,num_transforms=3,image_interpolation='linear'):0.2,
            tio.RandomGhosting(num_ghosts=3,axes=(0,1),intensity=0.3):0.2,
            #tio.RandomBiasField(coefficients=0.4,order=1):0.2,
            #tio.RandomNoise(std=0.1):0.2,
            tio.RandomBlur(std=0.6):0.2,
            #tio.RandomGamma((-0.1, 0.1)):0.2
            #tio.RandomAffine(scales=(0.9, 1.2),degrees=5,isotropic=True,image_interpolation='linear',):0.4
            #tio.RandomElasticDeformation(num_control_points=(7, 7, 7),locked_borders=2,):1
        } 
        im_msk_transform = {
            #tio.RandomNoise(std=0.03):0.2
            #tio.RandomGamma((- 0.3, 0.3)),
            tio.RandomAffine(scales=(0.9, 1.1),degrees=2,isotropic=True,image_interpolation='linear',):0.5,
        }
        transform1 = tio.Compose([
            tio.OneOf(im_msk_transform,p=0.3)
        ])
        transform2 = tio.Compose([
            tio.OneOf(spatial_transforms,p=0.8)
        ])
        
        transformed = transform1(image.permute(0,3,2,1)).permute(0,3,2,1)
        img_transformed = transform2(transformed[None,0,...].permute(0,3,2,1)).permute(0,3,2,1)
        return img_transformed[0,...],transformed[1:,...]

        #mask shape : CZYX
    def __len__(self):
        return len(self.data)

    

def post_process(pred):
    #NCDHW input
    classes = pred.shape[1]    
    #pred = F.softmax(pred,dim=1)
    maxvar = t.zeros_like(pred)
    for i in range(classes):
        maxvar[:,i,...] = t.argmax(pred,1)==i
    del pred
    transform = monai.transforms.KeepLargestConnectedComponent([1])
    
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

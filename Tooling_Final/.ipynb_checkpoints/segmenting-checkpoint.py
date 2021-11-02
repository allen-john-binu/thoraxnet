#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import os
from tqdm import tqdm
import torch as t
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from utils import *
from SegLargeOrgan import UnetModel
from SegSmallOrgan import UnetModel as UnetmodelSmall
t.backends.cudnn.enabled = True

Lung_R_weight = "/workstation/seenia/autocontouring/Thoracic_Autocontouring/manu/FinalWeights/@@RBLURUnet_3D_3level-32_Lung_R_epochs_120_trainsize_36_loss_dice[0.1,1.9]_trainloss_0.7951728105545044"
Lung_L_weight = "/workstation/seenia/autocontouring/Thoracic_Autocontouring/manu/FinalWeights/@@RBLURUnet_3D_3level-32_Lung_L_epochs_120_trainsize_36_loss_dice[0.1,1.9]_trainloss_0.9063127897679806"
spine_weight = "/workstation/seenia/autocontouring/Thoracic_Autocontouring/manu/3D-Model/smallOrgan/weight/SmallSpinal_80Epochs3.8146427534520626"
eso_weight = "/workstation/seenia/autocontouring/Thoracic_Autocontouring/manu/3D-Model/smallOrgan/weight/SmallEso_80Epoch5.67391224950552"
heart_weight = "/workstation/seenia/autocontouring/Thoracic_Autocontouring/manu/FinalWeights/@@RBLURUnet_3D_3level-32_Heart_epochs_160_trainsize_36_loss_dice[0.5,1.5]_trainloss_2.180488631129265"

mask_names = ['Lung_L','Lung_R','Heart','SpinalCord','Esophagus']

masks = {
    'Lung_L':Lung_L_weight,
     'Lung_R':Lung_R_weight,
     'Heart':heart_weight,
     'SpinalCord':spine_weight,
     'Esophagus':eso_weight
        }


# In[ ]:


def pad_mask(predmask,croplims):
    predmask = F.pad(predmask, [croplims[4], croplims[5], croplims[2], croplims[3] , croplims[0], croplims[1]])
    return predmask

def post_process_combined(pred,logits):
    maxvar = t.zeros_like(pred)
    for i in range(logits.shape[1]):
        maxvar[:,i,...] = t.argmax(logits,1)==i
    logits = maxvar
    pos = (t.sum(pred,axis=1)>1).reshape((pred.shape[0],1,pred.shape[2],pred.shape[3],pred.shape[4]))
    pos = t.repeat_interleave(pos,repeats = pred.shape[1],dim=1)

    pred = t.where(pos, logits, pred)
    return pred.cpu().numpy()


# In[ ]:
def segment(image,crops):
    ##import image into function
    #from monai.metrics import HausdorffDistanceMetric,SurfaceDistanceMetric

    final_pred = []
    #actual_masks=[]
    logits = []
    '''
    hd = HausdorffDistanceMetric(percentile=95,reduction="none",include_background=True)
    msd = SurfaceDistanceMetric(include_background=True)
    '''
    for organ in mask_names:
        print("Predicting : " + organ)

        croplimits = [crops[organ]["minz"],crops[organ]["maxz"],crops[organ]["minx"],crops[organ]["maxx"],crops[organ]["miny"],crops[organ]["maxy"]]
        padlimits = [croplimits[0],image.shape[0] - croplimits[1],croplimits[2],512-croplimits[3],croplimits[4],512-croplimits[5]]
        #######CHECK CROPPING LIMITS
        
        cropped_image = image[croplimits[0]:croplimits[1],croplimits[2]:croplimits[3],croplimits[4]:croplimits[5]]
        cropped_image = load_image(cropped_image)
        #print(cropped_image.shape)
        
        if organ in ['SpinalCord','Esophagus']:
            model = UnetmodelSmall(in_channels=1,out_classes=2)
        else:
            model = UnetModel(in_channels=1,out_classes=2)

        model.load_state_dict(t.load(masks[organ],map_location='cuda:0')) #Loading model weights for current organ
        model.eval()
        #pred = t.zeros_like(mask)
        #mask = mask.to('cuda:0')

        with t.no_grad():
            pred_mask = model(cropped_image.to('cuda:0'))
            pred_mask = F.softmax(pred_mask,dim=1)
            pred = post_process(pred_mask)            

            '''dices.append(DiceCoeff(mask[:,1:,...],pred[:,1:,...]).cpu().numpy())
            precisions.append(precision(mask[:,1:,...],pred[:,1:,...]).cpu().numpy())
            recalls.append(recall(mask[:,1:,...],pred[:,1:,...]).cpu().numpy())
            ious.append(iou(mask[:,1:,...],pred[:,1:,...]).cpu().numpy())
            hds.append(hd(mask[:,1:,...].permute(0,1,3,4,2),pred[:,1:,...].permute(0,1,3,4,2))[0])
            msds.append(msd(mask[:,1:,...].permute(0,1,3,4,2),pred[:,1:,...].permute(0,1,3,4,2))[0])'''

            #act_mask = np.load(actual_data_path+name+"/structure/"+organ+".npy").astype(np.uint8)
            final_pred.append(pad_mask(pred[:,1:,...],padlimits))
        
            logits.append(pad_mask(pred_mask[:,1:,...],padlimits))
            #actual_masks.append(act_mask.reshape((1,1,)+act_mask.shape))

            ###Check if it needs to be numpy or cpu

            if organ== mask_names[-1]:
                final_pred = t.cat(final_pred,axis=1).cpu()
                #actual_masks = np.concatenate(actual_masks,axis=1)
                logits = t.cat(logits,axis=1).cpu()



        #plt.figure()
        #plt.title(organ)
        #plt.boxplot(np.asarray(dices).reshape((12,1)),boxprops=dict(facecolor="#ff9966"),patch_artist=True,medianprops=dict(color="#000000"))
        #plt.show()
        #plt.savefig("/workstation/seenia/autocontouring/Thoracic_Autocontouring/manu/boxplots/"+organ+".png",dpi=200)
        '''
        mean_dice = np.mean(np.asarray(dices).reshape((12,1)),axis=0)
        mean_hd = np.mean(np.asarray(hds).reshape((12,1)),axis=0)
        mean_msd = np.mean(np.asarray(msds).reshape((12,1)),axis=0)
        mean_precision = np.mean(np.asarray(precisions).reshape((12,1)),axis=0)
        mean_recall = np.mean(np.asarray(recalls).reshape((12,1)),axis=0)
        mean_iou = np.mean(np.asarray(ious).reshape((12,1)),axis=0)

        print("Performance of "+organ+":")
        print("Mean Dice : "+ str(mean_dice[0]))
        print("Mean 95 percentile HD : "+ str(mean_hd[0]))
        print("Mean MSD : "+ str(mean_msd[0]))
        print("Mean Precision : "+ str(mean_precision[0]))
        print("Mean Recall : "+ str(mean_recall[0]))
        print("Mean IoU : "+ str(mean_iou[0]))
        print("")
        '''
        del model
    #print(final_pred.shape)
    #print(np.count_nonzero(np.sum(final_pred.numpy(),axis=1)>1))
    final_pred= post_process_combined(final_pred.to('cuda:0'),logits.to('cuda:0'))
    #print(np.count_nonzero(np.sum(final_pred,axis=1)>1))  

    return final_pred
# In[ ]:





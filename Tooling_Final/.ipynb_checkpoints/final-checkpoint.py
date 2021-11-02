import os, sys, glob
from scipy.ndimage import zoom
import numpy as np
import pydicom as pd
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
from os import path
from skimage.draw import polygon
from tqdm import tqdm
from cropping import calc_crop
import argparse
from segmenting import segment
import time
names = ['Lung_L','Lung_R','Heart','SpinalCord','Esophagus']
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

        
##Reading Data From Dicom

def normalize(im_input):
    
    minv = im_input.min()
    maxv = im_input.max()
    im_input = np.float32((im_input - minv)*1.0 / (1.0*(maxv - minv)))
    return im_input
    
    
def get_hu_values(image,slices):
    image = image.astype(np.int32)
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    image = np.array(image, dtype=np.int32)
    image[image < -1024] = -1024
    image[image>2000] = 2000
    image = normalize(image)
    return image
    
    
def downsample_img(image):
	downsampled = zoom(image, (0.25, 0.25, 0.25))
	downsampled = downsampled.transpose(2,0,1)
	return downsampled
    

def read_images_masks(patient):
    for subdir, dirs, files in os.walk(patient):
        dcms = glob.glob(os.path.join(subdir, "*.dcm"))
        if len(dcms) < 1:
            sys.exit("Expected multiple DICOM slices in source")
        elif len(dcms) >1:
            slices = [pd.read_file(dcm) for dcm in dcms]
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
            image = np.stack([s.pixel_array for s in slices], axis=-1)
        print("Image imported successfully")
        print("Preprocessing...", end="")
    image = get_hu_values(image,slices)
    print("Done\n")
    
    return image
    
def full_process():
    SRC_DATA = opt.source
    starttime = time.time()
    patient = opt.source
    #patients[1] ##check
    name_idx = len(SRC_DATA)
    try:
        image = read_images_masks(patient)
        print("Shape of Image: ",image.shape,"\n")
    except Exception as e:
        print("Encountered an error while preprocessing image")
        sys.exit(0)
    
    print("Cropping Images...")
    crop_input = downsample_img(image)
    image = image.transpose(2,0,1)
    
    #Predicting and cutting bounding box
    starttime = time.time()
    crop_values = calc_crop(crop_input)
    
    #Prediction
    print('Predicting masks... \n')
    final_mask = segment(image,crop_values)
    print("\nPrediction Completed\n")
    print("Total prediction time = ",time.time()-starttime)
    #Saving results
    print('Saving results... ')
    print("Final mask shape : ",final_mask[0,4,...].shape)
    np.save('/workstation/seenia/autocontouring/Thoracic_Autocontouring/data/TestCases/LCTSC-Test-S1-201/05-10-2004-RTRCCTTHORAX8F Adult-73548/1.000000-.simplified-65516/mask.npy',final_mask) 
    
    rtstruct = RTStructBuilder.create_new(dicom_series_path=opt.source)
    names = ['Lung_R','Lung_L','Heart','SpinalCord','Esophagus']
    colour = [[197,165,145],[197,165,145],[127,150,88],[253,135,192],[85,188,255]]
    with HiddenPrints():
        for organ,clr in zip(names,colour):

            result = final_mask[0,names.index(organ),...]
            result = result > 0
            result = result.transpose(1,2,0)
            rtstruct.add_roi(
                mask = result, 
                color = clr,
                name = organ
                )
    
    rtstruct.save(opt.dest+'final')
    print('Successfully saved predicted masks')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default= '/workstation/seenia/autocontouring/Thoracic_Autocontouring/data/TestCases/LCTSC-Test-S1-201/05-10-2004-RTRCCTTHORAX8F Adult-73548/0.000000-CT135227RespCT  3.0  B30f  50 Ex-15204', help='path to dicom series')  
    
    #parser.add_argument('--source', type=str, default='/workstation/seenia/autocontouring/Thoracic_Autocontouring/manu/LCTSC-Test-S3-202/10-21-2003-LTLUNG UP DIBH-46036/1.000000-79554', help='path to dicom series')
    parser.add_argument('--dest', type=str, default='/workstation/seenia/autocontouring/Thoracic_Autocontouring/data/TestCases/LCTSC-Test-S1-201/05-10-2004-RTRCCTTHORAX8F Adult-73548/1.000000-.simplified-65516', help='path to destination')
    #parser.add_argument('--organs', type=str, default='lhse', help='specific organs')
    
    #parser.add_argument('--dest', type=str, default='/workstation/seenia/autocontouring/Thoracic_Autocontouring/manu/', help='path to destination')
    #parser.add_argument('--organs', type=str, default='lhse', help='specific organs')
    
    opt = parser.parse_args()
    flag = 0
    
    if(path.exists(path.exists(opt.source) & path.exists(opt.dest))):
        try:
            demo = RTStructBuilder.create_new(dicom_series_path=opt.source)
            flag = 1
        except:
            print("No DICOM series found in input path")
    else:
        print ("Source File exists:" + str(path.exists(opt.source)))
        print ("Destination File exists:" + str(path.exists(opt.dest)))
    
    if(flag == 1):
        full_process()
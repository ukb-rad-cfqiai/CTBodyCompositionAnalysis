#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Clinic for Diagnositic and Interventional Radiology, University Hospital Bonn, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import  os, cv2, pickle
import numpy as np
import nibabel as nib
from skimage.transform import resize

CLASS_MUSCLE = 1
CLASS_VISCERAL_FAT = 2
CLASS_SUBCUTAN_FAT = 3
CLASS_MUSCLE_FAT = 4
CLASS_MUSCLE_LOW = 5
WINDOW_MUSCLE_LOW = [ -29,  29  ]
WINDOW_MUSCLE_HIGH  = [ 29,  100  ] 
WINDOW_FAT = [ -190 , -30  ]

def write_nii(im, affine, vox_spacing, out_path, as_uint8=False):  
    if len(im.shape) == 2: im = np.expand_dims(im, -1)
    if len(vox_spacing) == 2: vox_spacing = list(vox_spacing)+[1]
    niiFile = nib.Nifti1Image(im.astype(np.uint8) if as_uint8 else im.astype(float), affine )
    niiFile.header['pixdim'][1:4] = vox_spacing
    niiFile.header.set_sform(affine,code=1)
    niiFile.header.set_qform(affine,code=1)
    niiFile.header.set_zooms( vox_spacing )
    nib.save( niiFile, out_path )
    
def imfill(mask, setLeftRightBorderToOne = False):
       # imfill in cv2 https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
       h, w = mask.shape[:2]
       mask_filled = np.zeros((h+2, w+2), np.uint8)
       mask_copied = mask.copy()
       if setLeftRightBorderToOne:
           mask_copied[3:,0] = 1; mask_copied[:h-3,w-1] = 1 
           mask_copied[0:3,0] = 0; mask_copied[h-3:h,w-1] = 0;
       cv2.floodFill(mask_copied, mask_filled, (0,0), 1);
       # cv2.floodFill(mask_copied, mask_filled, (h-1,w-1), 1);
       cv2.floodFill(mask_copied, mask_filled, (w-1,h-1), 1);
       mask_filled_inv = np.logical_not(mask_copied.astype(bool))
       mask = mask | mask_filled_inv
       if setLeftRightBorderToOne:
           mask[:,0] = 0; mask[0,3:] = 0 
       return mask
   
def getLargestContour(mask, number_contours=1):
    centers, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_centers = sorted(centers, key=cv2.contourArea, reverse=True)
    mask_out = np.zeros(mask.shape, np.uint8)
    for n in range(number_contours):
        if n > len(sorted_centers)-1: break
        cv2.drawContours(mask_out, sorted_centers, n, 1, cv2.FILLED) #[cnts_sorted[n]]
    return cv2.bitwise_and(mask, mask_out)


def getHalfXandThirdYOfBody(image):
        mask = np.uint8(image > -400)
        kernel = np.uint8(np.ones((5,5)))
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask = cv2.dilate(mask,kernel,iterations = 1)
        mask = imfill(mask)
        mask = cv2.erode(mask,kernel,iterations = 2)
        mask = getLargestContour(mask)
        mask = cv2.dilate(mask,kernel,iterations = 2)
        indicies = np.where(mask==1)
        
        start_body_x = int(np.min(indicies[1]))
        end_body_x = int(np.max(indicies[1]))
        mid_body_x = int(np.floor(start_body_x + (end_body_x-start_body_x)/2))
        mask[:,mid_body_x:]= 0
        start_body_y = int(np.min(indicies[0]))
        end_body_y = int(np.max(indicies[0]))
        start_midThird_body_y = int(np.floor(start_body_y + (end_body_y-start_body_y)/3))
        end_midThird_body_y = int(np.floor(start_body_y + 2*(end_body_y-start_body_y)/3))
        mask[0:start_midThird_body_y,:]= 0
        mask[end_midThird_body_y:,:]= 0
        not_mask = np.logical_not(mask>0)
        image[not_mask] = np.min(image)
        return image

def printImageLabelOverlay(imageIn,labelIn,savePath, printImageWithoutOverlay=False, num_classes=4):
    
    image = np.squeeze(np.copy(imageIn))
    label = np.squeeze(np.copy(labelIn)).astype(float)
    label[label==CLASS_MUSCLE_FAT] = 2.4
    label[label==CLASS_MUSCLE_LOW] = 1.3
    image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
    
    label = np.array(label * 255 / (num_classes-1), dtype = np.uint8)
    heatmapImage = cv2.applyColorMap(label, cv2.COLORMAP_JET).astype(np.uint8)
    
    if len(np.shape(image)) == 2:
        new_image =  np.zeros( np.shape(image)+(3,)).astype(float)
        new_image[:,:,0] = image
        new_image[:,:,1] = image 
        new_image[:,:,2] = image 
        image = new_image
    
    maskLabel = np.zeros( np.shape(image)).astype(bool)
    maskLabel[:,:,0] = label > 0
    maskLabel[:,:,1] = maskLabel[:,:,0] 
    maskLabel[:,:,2] = maskLabel[:,:,0] 
    maskBg = np.zeros(np.shape(image)).astype(bool)
    maskBg[maskLabel==False]=True
    heatmapImage = cv2.addWeighted(heatmapImage, 0.6, image , 0.4, 0, dtype = cv2.CV_8UC1)
    heatmapImage[maskBg] = image[maskBg]
    cv2.imwrite(savePath,  np.flip(np.swapaxes(heatmapImage,0,1),0))
    
    if printImageWithoutOverlay:
        cv2.imwrite(savePath.replace("_segm.","_image."),  np.flip(np.swapaxes(image,0,1),0))

def printImageLabelOverlay_FMF(imageIn,labelIn,savePath, printImageWithoutOverlay=False):
    
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lookUpTable_blueFMF.pkl'), 'rb') as f:
        lookUpTable_blueFMF = pickle.load(f)
    B = lookUpTable_blueFMF[:,0]
    G = lookUpTable_blueFMF[:,1]
    R = lookUpTable_blueFMF[:,2]
    N = len(R)
    X = np.arange(0, 4*N, 4)
    X_new = np.arange(4*N)
    R = np.interp(X_new, X, R) 
    G = np.interp(X_new, X, G) 
    B = np.interp(X_new, X, B) 

    lookUpTable_blueFMF_new = np.dstack( (R, G, B) ).astype(np.uint8)
    image = np.squeeze(np.copy(imageIn))
    label = np.squeeze(np.copy(labelIn)).astype(float)
    label[label==1] = 1/255
    label[label==2] = 1
    image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    image.astype(np.uint8)
    label = np.array(label * 255 , dtype = np.uint8)
    heatmapImage = np.zeros(  (np.shape(label)+(3,) ))
    heatmapImage[:,:,0] = cv2.LUT(label, R)
    heatmapImage[:,:,1] = cv2.LUT(label, G)
    heatmapImage[:,:,2] = cv2.LUT(label, B)
    heatmapImage = heatmapImage.astype(np.uint8)
    
    if len(np.shape(image)) == 2:
        new_image =  np.zeros( np.shape(image)+(3,)).astype(float)
        new_image[:,:,0] = image
        new_image[:,:,1] = image 
        new_image[:,:,2] = image 
        image = new_image

    maskLabel = np.zeros( np.shape(image)).astype(bool)
    maskLabel[:,:,0] = label > 0
    maskLabel[:,:,1] = maskLabel[:,:,0] 
    maskLabel[:,:,2] = maskLabel[:,:,0] 
    maskBg = np.zeros(np.shape(image)).astype(bool)
    maskBg[maskLabel==False]=True
    heatmapImage[maskBg] = image[maskBg]
    cv2.imwrite(savePath, np.flip(np.swapaxes(heatmapImage,0,1),0))
    
    if printImageWithoutOverlay:
        cv2.imwrite(savePath.replace(".","_orig."),  np.flip(np.swapaxes(image,0,1),0))

def createMuscleQualityLabel( image, label):
    image = np.squeeze(np.copy(image)).astype(np.float32)
    label = np.squeeze(np.copy(label)) 
    muscleQualityLabel = np.copy(label)
    muscleQualityLabel[ ((label == CLASS_MUSCLE) & (image >= WINDOW_FAT[0])) & (image <= WINDOW_FAT[1]) ] = CLASS_MUSCLE_FAT
    muscleQualityLabel[ ((label == CLASS_MUSCLE) & (image >= WINDOW_MUSCLE_LOW[0])) & (image <= WINDOW_MUSCLE_LOW[1]) ] = CLASS_MUSCLE_LOW
    return muscleQualityLabel

def createdWindowedAndErodedVisceralMasks( image, label):
    image = np.squeeze(np.copy(image)).astype(np.float32)
    label = np.squeeze(np.copy(label))
    windowedLabel_visc = np.zeros(np.shape(label), dtype=bool)
    windowedLabel_visc[ ((label == CLASS_VISCERAL_FAT) & (image >= WINDOW_FAT[0])) & (image <= WINDOW_FAT[1]) ] = True
    windowedLabel_visc_eroded = cv2.erode(windowedLabel_visc.astype(np.uint8), kernel = np.uint8(np.ones((3,3))), iterations = 2).astype(bool)
    return windowedLabel_visc, windowedLabel_visc_eroded







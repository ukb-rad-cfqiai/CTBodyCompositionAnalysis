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

import numpy as np
import scipy.ndimage as ndimage
import cv2
import matplotlib.pyplot as plt


def plotData(image,coord_gt=None,label=None,coord_pred=None,pred=None,
             color_line_gt='lime',color_line_pred='cyan',lowestValue=None,
             highestValue=None,plane='sagittal',resolutionsRatioZ = 1, f=None,plotSegm=True):
    
    if not f:
        f = plt.figure()
    
    if label is None:
        label = np.zeros(image.shape)
    
    if pred is None:
        pred = np.zeros(image.shape)
    
    exist_coord_gt = True
    if coord_gt is None:
        exist_coord_gt = False
        if label is None or np.max(label)==0:
            coord_gt = np.round(np.array(image.shape)/2).astype(int)
        else:
            coord_gt = np.round(ndimage.center_of_mass(label)).astype(int)
            
    exist_coord_pred = True
    if coord_pred is None:
        exist_coord_pred = False
        coord_pred = np.round(np.array(image.shape)/2).astype(int)

    image = np.squeeze(image)
    label = np.squeeze(label)
    pred = np.squeeze(pred)

    if len(np.shape(image)) == 4: #take first channel
        image = image[0,:]
     
    if lowestValue:
        image[image<lowestValue] = lowestValue
    if highestValue:
        image[image>highestValue] = highestValue
     
    if plane == 'axial':
        image_plot = image[:,:,coord_gt[2]]
        label_plot = label[:,:,coord_gt[2]]
        pred_plot = pred[:,:,coord_gt[2]]
    elif plane == 'coronal':
        image_plot = image[:,coord_gt[1],:]
        label_plot = label[:,coord_gt[1],:]
        pred_plot = pred[:,coord_gt[1],:]

    elif plane == 'sagittal':
        image_plot = image[coord_gt[0],:,:]
        label_plot = label[coord_gt[0],:,:]
        pred_plot = pred[coord_gt[0],:,:]
   
    image_plot = np.flip(image_plot)  
    label_plot = np.flip(label_plot)
    pred_plot = np.flip(pred_plot)
  
    image_plot = image_plot.transpose((1, 0))  
    label_plot = label_plot.transpose((1, 0))
    pred_plot = pred_plot.transpose((1, 0))

    image_shape = np.asarray(np.shape(image_plot))
    z_coord_plot_gt = image_shape[0]-coord_gt[2]-1  
    z_coord_plot_pred = image_shape[0]-coord_pred[2]-1  
    
    if not resolutionsRatioZ == 1:
        if plane == 'sagittal' or plane == 'coronal':
            image_shape = np.asarray(np.shape(image_plot))
            image_shape[0] *= resolutionsRatioZ
            z_coord_plot_gt *= resolutionsRatioZ
            z_coord_plot_pred *= resolutionsRatioZ
            width = image_shape[1]
            height = image_shape[0]
            dim = [width,height]
            image_plot = cv2.resize(image_plot,tuple(dim))
            label_plot = cv2.resize(label_plot,tuple(dim),interpolation=cv2.INTER_NEAREST)
            pred_plot = cv2.resize(pred_plot,tuple(dim),interpolation=cv2.INTER_NEAREST)
   
    
    if plotSegm:
        imageForHeatmap = pred_plot.astype(float)
        imageForHeatmap[ np.logical_and( pred_plot==0, label_plot>0 ) ] = 0.75
        imageForHeatmap[ np.logical_and( np.equal( pred_plot, label_plot ), label_plot>0 ) ] = 0.5
        
        image_plot = cv2.normalize(image_plot, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        image_plot = image_plot.astype(np.uint8)        
        
        im = np.zeros( np.shape(image_plot) +(3,)).astype(np.uint8)
        im[:,:,0] = image_plot
        im[:,:,1] = im[:,:,0]
        im[:,:,2] = im[:,:,0]
        
        maskLabel = np.zeros( np.shape(im)).astype(bool)
        maskLabel[:,:,0] = imageForHeatmap > 0
        maskLabel[:,:,1] = maskLabel[:,:,0]
        maskLabel[:,:,2] = maskLabel[:,:,0]
    
        imageForHeatmap = imageForHeatmap * 255.0
        imageForHeatmap = imageForHeatmap.astype(np.uint8)
        heatmapImage = cv2.applyColorMap(imageForHeatmap, cv2.COLORMAP_JET).astype(float)
       
        tmp = np.copy(heatmapImage[:,:,0])
        heatmapImage[:,:,0] = heatmapImage[:,:,2]
        heatmapImage[:,:,2] = tmp
        heatmapImage = heatmapImage.astype(np.uint8)
        
        maskBg = np.zeros(np.shape(im)).astype(bool)
        maskBg[maskLabel==False]=True
        heatmapImage = cv2.addWeighted(heatmapImage, 0.4, im , 0.6, 0, dtype = cv2.CV_8UC1)
        heatmapImage[maskBg] = im[maskBg]
        
        if not plane=='axial':
            if exist_coord_gt:
                plt.axhline(z_coord_plot_gt,color=color_line_gt)
            if exist_coord_pred:
                plt.axhline(z_coord_plot_pred,color=color_line_pred)
           
        plt.imshow(heatmapImage).set_cmap('jet')
        plt.axis('off')
    
    else:
        plt.imshow(image_plot,cmap='gray')
        plt.axis('off')

    return f
    
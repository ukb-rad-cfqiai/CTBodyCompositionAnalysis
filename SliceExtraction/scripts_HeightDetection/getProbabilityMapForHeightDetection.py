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
# only relevant for training
def getProbabilityMap(image,coord,regression=False,sigmaXY=8,sigmaZ=1.6):
    
    if coord:
        containsL3L4 = True
    else:
        containsL3L4 = False
        
    if containsL3L4:
        sizeImage = image.shape
        ii,jj,kk = np.mgrid[0:sizeImage[0],0:sizeImage[1],0:sizeImage[2]]
        indices = [ii.flatten(order='F'),jj.flatten(order='F'),kk.flatten(order='F')]
        
        distanceToVertebraeXYPlane = np.linalg.norm([np.array(indices[0])-coord[0],np.array(indices[1])-coord[1]],axis=0)**2
        distanceToVertebraeZPlane = np.linalg.norm([np.array(indices[2])-coord[2]],axis=0)**2
        
        probMap = (1/(sigmaXY*np.sqrt(2*np.pi))*np.exp(-distanceToVertebraeXYPlane/(2*sigmaXY**2)))*(1/(sigmaZ*np.sqrt(2*np.pi))*np.exp(-distanceToVertebraeZPlane/(2*sigmaZ**2)))
        probMap = np.single(np.reshape(probMap,sizeImage,order='F'))
        
        if not regression:
            # segmentation task, label consists of zeros and ones
            newLabel = np.zeros(sizeImage)
            threshVal = 0.05 * max(probMap.flatten())
            newLabel[np.where(probMap > threshVal)]= 1.0
            probMap = np.uint8(newLabel)
            
    return probMap
            

        
        

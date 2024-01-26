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

import sys
import os
parent_directory = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])
sys.path.append(parent_directory)
import numpy as np
import nibabel as nib
from utils.plotData import plotData
import scipy.ndimage as ndimage
import shutil
import matplotlib.pyplot as plt


def getCoordinateFromPrediction(cur_result):
    l3l4Coordinate = np.round(ndimage.center_of_mass(cur_result)).astype(int)
    return l3l4Coordinate
    
def comparePredCoordWithGT(l3l4_pred,l3l4_gt,spacing_z):
    euclid_dist = np.linalg.norm(np.array(l3l4_pred)-np.array(l3l4_gt))
    diff_z = abs(l3l4_pred[2]-l3l4_gt[2])     
    diff_z_in_mm = np.round(diff_z*spacing_z,4)

    return euclid_dist, diff_z, diff_z_in_mm

def predictCase(pred,outputPath,rawData,groundTruth,orgPatientNames,newPatientNames,gtName,visualization=False,saveL3L4Slice=True,saveOriginal=False,cropExtremeties=False):

    resultCenters = []
    resultCenters_OrgSpacing = []
     
    if '/validation' in pred:
        predValidation=True
    else:
        predValidation=False
        
    indxTask = pred.find('Task')
    taskName = pred[indxTask:]
    indxFilesep = [i for i in range(0,len(taskName)) if taskName[i]=='/']
    taskName = taskName[0:indxFilesep[0]]
    
    nnUNet_path = os.path.expandvars('$nnUNet_raw_data_base')
    
    if saveL3L4Slice:
        l3l4SlicePath = f'{outputPath}/L3L4Slice'

    if visualization:
        sagittalPath = f'{nnUNet_path}/Plots/{taskName}/Pred/sagittal' 
        coronalPath = f'{nnUNet_path}/Plots/{taskName}/Pred/coronal' 
        axialPath = f'{nnUNet_path}/Plots/{taskName}/Pred/axial' 


    # ----------- Find corresponding raw pred and ground truth ------------  
    curPatName = os.path.basename(pred).split('.')[0]
    curRawData = rawData[[i for i in range(0,len(rawData)) if curPatName in rawData[i]][0]]
    if not curRawData:
        raise(f'ERROR! Could not find raw image for {curPatName}')
    curGT = [i for i in groundTruth if curPatName in i]
    if curGT:
        if len(curGT)>1:
            raise(f'ERROR! Found more than one Ground Truth for patient {curPatName}')
        curGT = curGT[0]
        existGT = True
    else:
        existGT = False
    
    print(f'predict patient {curPatName}\n')
    
    nii_raw = nib.load(curRawData)
    nii_pred = nib.load(pred)
    nii_pred_image = nii_pred.get_fdata()

    curSpacing = nii_pred.header.get_zooms()
    
    if existGT:
        nii_gt = nib.load(curGT)
    # ---------------------------------------------------------------------
    cur_filename = nii_raw.get_filename()
    
    existWarning = False
    if os.path.exists(f'{cur_filename.split(".")[0]}_Warning.txt'):
        existWarning = True
      
    
    # ----------------------- Compare With GT -----------------------------
    if existGT:
        cur_filename = nii_raw.get_filename()
         
        coord_path = f'{cur_filename.split(".")[0]}_{gtName}' 
        gtName_resizing = gtName.replace('.txt','_AfterResizing.txt')
        coord_path_resizing = f'{cur_filename.split(".")[0]}_{gtName_resizing}' 
        
        existCoord = True
        if os.path.exists(coord_path) and not os.path.exists(coord_path_resizing):
            coord = open(coord_path)
            coord = coord.read().split(',')
            if 'NaN' in coord:
                coord = [np.NaN,np.NaN,np.NaN]
            else:
                coord = [int(coord[0]),int(coord[1]),int(coord[2])]
        elif os.path.exists(coord_path_resizing):
            coord = open(coord_path_resizing)
            coord = coord.read().split('\n')[1].split(':')[1].split(',')
            if 'NaN' in coord:
                coord = [np.NaN,np.NaN,np.NaN]
                coord_orgSpacing = [np.NaN,np.NaN,np.NaN]
            else:
                coord = [int(coord[0]),int(coord[1]),int(coord[2])]   
                coord_orgSpacing = open(coord_path_resizing)
                coord_orgSpacing = coord_orgSpacing.read().split('\n')[0].split(':')[1].split(',')
                coord_orgSpacing = [int(coord_orgSpacing[0]),int(coord_orgSpacing[1]),int(coord_orgSpacing[2])]
        else:
             print(f'Could not find text file with original coordinates for patient {curPatName}')
             existCoord = False
        
        if np.max(nii_pred_image.flatten()): #l3l4Slice in prediction
            l3l4_pred  = getCoordinateFromPrediction(nii_pred_image)
            euclid_dist,diff_z,diff_z_in_mm = comparePredCoordWithGT(l3l4_pred,coord,curSpacing[2])
            resultCenters = [curPatName,coord,l3l4_pred,euclid_dist, diff_z, diff_z_in_mm, existWarning]
        # ---------------------------------------------------------------------
               
        else: # no l3l4Slice in prediction
            l3l4_pred = [np.NaN,np.NaN,np.NaN]
            resultCenters = [curPatName,coord,l3l4_pred,np.NaN,np.NaN,np.NaN, existWarning]
       
        if visualization:
            resolutionsRatioZ = nii_pred.header.get_zooms()[2]/nii_pred.header.get_zooms()[1]
            
            l3l4_pred_forPlot = l3l4_pred
            if np.NaN in l3l4_pred_forPlot:
                l3l4_pred_forPlot = None
            
            if existCoord and np.NaN not in coord:
                for cur_plane in ['axial','sagittal','coronal']:
                    fig = plotData(image=nii_raw.get_fdata(),coord_gt=coord,label=nii_gt.get_fdata(),
                                   coord_pred=l3l4_pred_forPlot,pred=nii_pred_image,
                                   lowestValue=-400,highestValue=600,plane=cur_plane,
                                   resolutionsRatioZ=resolutionsRatioZ)
                    pathName = f'{cur_plane}Path'
                    fig.savefig(f'{eval(pathName)}/{curPatName}_{cur_plane}.png',dpi=600)
                    plt.close(fig)
            else:
                for cur_plane in ['axial','sagittal','coronal']:
                    fig = plotData(image=nii_raw.get_fdata(),coord_gt=None,label=nii_gt.get_fdata(),
                                   coord_pred=l3l4_pred_forPlot,pred=nii_pred_image,
                                   lowestValue=-400,highestValue=600,plane=cur_plane,
                                   resolutionsRatioZ=resolutionsRatioZ)
                    pathName = f'{cur_plane}Path'
                    fig.savefig(f'{eval(pathName)}/{curPatName}_{cur_plane}.png',dpi=600)
                    plt.close(fig)

    else: # doesn't exist GT (for examlpe in unlabeled test cases)
        if np.max(nii_pred_image.flatten()): #l3l4Slice in prediction
            l3l4_pred  = getCoordinateFromPrediction(nii_pred_image)
        else:
            l3l4_pred = [np.NaN,np.NaN,np.NaN]
        resultCenters = [curPatName,[np.NaN,np.NaN,np.NaN],l3l4_pred,np.NaN,np.NaN,np.NaN, existWarning]
        
        if visualization:
            resolutionsRatioZ = nii_pred.header.get_zooms()[2]/nii_pred.header.get_zooms()[1]
            l3l4_pred_forPlot = l3l4_pred
            if np.NaN in l3l4_pred_forPlot:
                l3l4_pred_forPlot = None
            
            for cur_plane in ['axial','sagittal','coronal']:
                fig = plotData(image=nii_raw.get_fdata(),coord_gt=None,label=None,
                               coord_pred=l3l4_pred_forPlot,pred=nii_pred_image,
                               lowestValue=-400,highestValue=600,plane=cur_plane,
                               resolutionsRatioZ=resolutionsRatioZ)
                pathName = f'{cur_plane}Path'
                fig.savefig(f'{eval(pathName)}/{curPatName}_{cur_plane}.png',dpi=600)
                plt.close(fig)
        
 
    # save predictions as .txt 
    cur_filename = nii_raw.get_filename()
            
    scalingFactors_path_1 = f'{cur_filename.split(".")[0]}_ScalingFactors.txt'
    scalingFactors_path_2 = f'{cur_filename.split(".")[0]}_{gtName.split(".")[0]}_AfterResizing.txt'
    
    changeVoxelSpacing = True
    if os.path.exists(scalingFactors_path_1):
        f = open(scalingFactors_path_1,"r")
        scalingFactorsString = f.read().split(':')[1].split(',')
        scalingFactors = [float(scalingFactorsString[0]),float(scalingFactorsString[1]),float(scalingFactorsString[2])]    
        f.close()
    elif os.path.exists(scalingFactors_path_2):
        f = open(scalingFactors_path_2,"r")
        scalingFactorsString = f.read().splitlines()[2].split(':')[1].split(',')
        scalingFactors = [float(scalingFactorsString[0]),float(scalingFactorsString[1]),float(scalingFactorsString[2])]
        f.close()
    else:
        changeVoxelSpacing = False
    
    txt_path = f'{cur_filename.split(".")[0]}_{gtName.split(".")[0]}_Prediction.txt'
    f = open(txt_path,"w")
    if changeVoxelSpacing:
        if any(np.isnan(l3l4_pred)):
            l3l4_pred_orgSpacing =  [np.NaN,np.NaN,np.NaN]
        else:
            l3l4_pred_orgSpacing = [int(np.round(l3l4_pred[i]*scalingFactors[i])) for i in range(0,len(scalingFactors))]
        f.writelines("PredictedCoordinate: %s,%s,%s\n" %(l3l4_pred[0],l3l4_pred[1],l3l4_pred[2]))
        f.writelines("PredictedCoordinate_OrgSpacing: %s,%s,%s" %(l3l4_pred_orgSpacing[0],l3l4_pred_orgSpacing[1],l3l4_pred_orgSpacing[2]))
        f.close()
        if existGT:
            org_spacing_z = curSpacing[2]/scalingFactors[2]
            euclid_dist,diff_z,diff_z_in_mm = comparePredCoordWithGT(l3l4_pred_orgSpacing, coord_orgSpacing, org_spacing_z)
            resultCenters_OrgSpacing = [curPatName,coord_orgSpacing,l3l4_pred_orgSpacing,euclid_dist, diff_z, diff_z_in_mm, existWarning]
        else:
            resultCenters_OrgSpacing = [curPatName,[np.NaN,np.NaN,np.NaN],l3l4_pred_orgSpacing,np.NaN,np.NaN,np.NaN, existWarning]
        
    else:
        f.writelines("PredictedCoordinate: %s,%s,%s" %(l3l4_pred[0],l3l4_pred[1],l3l4_pred[2]))
        f.close()
    
    if not predValidation: # one can only save l3/l4 slice images for test pred
        if saveL3L4Slice:
            if saveOriginal:
                if not any(np.isnan(l3l4_pred)):                    
                    curPatName_Modality = os.path.basename(cur_filename)
                    curOrgPatientPath = [orgPatientNames[x] for x in range(len(orgPatientNames)) if os.path.basename(newPatientNames[x]) == curPatName_Modality]
                      
                    if len(curOrgPatientPath) == 1:
                        curOrgPatientPath = curOrgPatientPath[0]
                    elif len(curOrgPatientPath) > 1:
                        raise(f'ERROR! Found more than one corresponding original paths for patient {curPatName}')
                    else:
                        raise(f'ERROR! Original path was not found for patient {curPatName}')
                        
                    orgPatName = os.path.basename(curOrgPatientPath).split('.')[0]
                    orgStudyFolder = os.sep.join(os.path.dirname(curOrgPatientPath).split(os.sep)[0:-2])
                    saveOrgPatientPath = os.path.join(orgStudyFolder,'nnUNet_L3L4Slice_Results',orgPatName)
                    if not os.path.exists(saveOrgPatientPath):
                        os.makedirs(saveOrgPatientPath)
                     
                   
                    if os.path.exists(curOrgPatientPath):
                        nii_org = nib.load(curOrgPatientPath)
                        
                        if changeVoxelSpacing:
                            l3l4Slice = nib.Nifti1Image(nii_org.get_fdata()[:,:,l3l4_pred_orgSpacing[2]], nii_org.affine)
                        else:
                            l3l4Slice = nib.Nifti1Image(nii_org.get_fdata()[:,:,l3l4_pred[2]], nii_org.affine)
                        
                        if nii_org.header.extensions:  
                            l3l4Slice.header.extensions = nii_org.header.extensions
                            
                        nib.save(l3l4Slice,os.path.join(saveOrgPatientPath,f"{orgPatName}_L3L4Slice.nii.gz"))

                    if visualization:
                        # save predicted l3/l4 slice of image with original spacing
                        l3l4_pred_forPlot = l3l4_pred_orgSpacing
                        if np.NaN in l3l4_pred_forPlot:
                            l3l4_pred_forPlot = None
                        sagittalOrgPath = os.path.join(saveOrgPatientPath,f"{orgPatName}_pred.png")
                        resolutionsRatioZ = nii_org.header.get_zooms()[2]/nii_org.header.get_zooms()[1]
                        fig = plotData(image=nii_org.get_fdata(),coord_gt=None,label=None,
                                       coord_pred=l3l4_pred_forPlot,pred=None,
                                       lowestValue=-400,highestValue=600,plane='sagittal',
                                       resolutionsRatioZ=resolutionsRatioZ)
                    
                        fig.savefig(sagittalOrgPath,dpi=600)
                        plt.close(fig)

                        
                        if existGT:
                            coord_orgSpacing_forPlot = coord_orgSpacing
                            if np.NaN in coord_orgSpacing_forPlot:
                                coord_orgSpacing_forPlot = None
                            sagittalOrgPath = os.path.join(saveOrgPatientPath,f"{orgPatName}_pred_gt.png")
                            fig = plotData(image=nii_org.get_fdata(),coord_gt=coord_orgSpacing_forPlot,label=None,
                                           coord_pred=l3l4_pred_forPlot,pred=None,
                                           lowestValue=-400,highestValue=600,plane='sagittal',
                                           resolutionsRatioZ=resolutionsRatioZ)
                        
                            fig.savefig(sagittalOrgPath,dpi=600)
                            plt.close(fig)

                           
            if not any(np.isnan(l3l4_pred)):
                l3l4Slice = nib.Nifti1Image(nii_raw.get_fdata()[:,:,l3l4_pred[2]], nii_raw.affine)   
                if nii_raw.header.extensions:  
                    l3l4Slice.header.extensions = nii_raw.header.extensions   
                nib.save(l3l4Slice,f'{l3l4SlicePath}/{curPatName}_L3L4Slice.nii.gz')
    
    if saveOriginal: #save to original pred path in original and nnUNet spacing
        curPatName_Modality = os.path.basename(cur_filename)
        curOrgPatientPath = [orgPatientNames[x] for x in range(len(orgPatientNames)) if os.path.basename(newPatientNames[x]) == curPatName_Modality]
         
        if len(curOrgPatientPath) == 1:
            curOrgPatientPath = curOrgPatientPath[0]
        elif len(curOrgPatientPath) > 1:
            raise(f'ERROR! Found more than one corresponding original paths for patient {curPatName}')
        else:
            raise(f'ERROR! Original path was not found for patient {curPatName}')
        
        orgPatName = os.path.basename(curOrgPatientPath).split('.')[0]
        orgStudyFolder = os.sep.join(os.path.dirname(curOrgPatientPath).split(os.sep)[0:-2])
        saveOrgPatientPath = os.path.join(orgStudyFolder,'nnUNet_L3L4Slice_Results',orgPatName)
        
        if not os.path.exists(saveOrgPatientPath):
            os.makedirs(saveOrgPatientPath)
        
        sagittalOrgPath = os.path.join(saveOrgPatientPath,f"{orgPatName}_pred.png")

        txt_path_coordOrgSpacing = os.path.join(saveOrgPatientPath,f'{orgPatName}_Prediction_orgSpacing.txt')
        f = open(txt_path_coordOrgSpacing,"w")
        
        txt_path_coordNewSpacing = os.path.join(saveOrgPatientPath,f'{orgPatName}_Prediction_nnUNetSpacing.txt')
        f2 = open(txt_path_coordNewSpacing,"w")
        
        txt_path_orgSpacing  = os.path.join(saveOrgPatientPath,f'{orgPatName}_orgSpacing.txt')
        f3 = open(txt_path_orgSpacing,"w")
        
        if os.path.exists(curOrgPatientPath):
            orgImage = nib.load(curOrgPatientPath)
            orgSpacing = orgImage.header.get_zooms() 
            f3.writelines("%s,%s,%s" %(orgSpacing[0],orgSpacing[1],orgSpacing[2]))
            f3.close()
            if changeVoxelSpacing:
                l3l4_pred_orgSpacing = [np.round(l3l4_pred[i]*scalingFactors[i]) for i in range(0,len(scalingFactors))]
                f.writelines("%s,%s,%s" %(l3l4_pred_orgSpacing[0],l3l4_pred_orgSpacing[1],l3l4_pred_orgSpacing[2]))
                f.close()
                f2.writelines("%s,%s,%s" %(l3l4_pred[0],l3l4_pred[1],l3l4_pred[2]))
                f2.close()
            else:  
                f.writelines("%s,%s,%s" %(l3l4_pred[0],l3l4_pred[1],l3l4_pred[2]))
                f.close()
                
            if os.path.exists(f'{cur_filename.split(".")[0]}_Warning.txt'):
                txt_path_warning = os.path.join(saveOrgPatientPath,f"{orgPatName}_Warning.txt")
                shutil.copy(f'{cur_filename.split(".")[0]}_Warning.txt',txt_path_warning)           
                
    return resultCenters,resultCenters_OrgSpacing


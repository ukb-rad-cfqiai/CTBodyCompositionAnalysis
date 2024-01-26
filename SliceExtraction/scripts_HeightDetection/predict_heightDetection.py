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
from scripts_HeightDetection.predict_singleCase_HD import predictCase
import multiprocessing as mp
import pandas as pd


def getRawDataAndGT(data):
    
    if '/validation' in data[0]:
        predValidation=True
    else:
        predValidation=False
        
    indxTask = data[0].find('Task')
    taskName = data[0][indxTask:]
    indxFilesep = [i for i in range(0,len(taskName)) if taskName[i]=='/']
    taskName = taskName[0:indxFilesep[0]]
    
    nnUNet_path = os.path.expandvars('$nnUNet_raw_data_base') 
    rawDataDir = "%s/nnUNet_raw_data/%s" %(nnUNet_path,taskName)
    
    if predValidation:
        imagesDir="%s/imagesTr" %rawDataDir
        labelsDir = "%s/labelsTr" %rawDataDir
    else:
        imagesDir = "%s/imagesTs" %rawDataDir
        labelsDir = "%s/labelsTs" %rawDataDir
    
    raw_data = []
    for root_path, dirs, files in os.walk(imagesDir):
        for name in files:
            if name.endswith(".nii") or name.endswith(".nii.gz"):
                raw_data.append(os.path.join(root_path,name))
    raw_data.sort()
           
    groundTruth = []
    for root_path, dirs, files in os.walk(labelsDir):
       for name in files:
           if name.endswith(".nii") or name.endswith(".nii.gz"):
               groundTruth.append(os.path.join(root_path,name))
    groundTruth.sort()  
    
    return raw_data, groundTruth

def saveResults(results,outputPath, saveOriginal, orgPatientNames, newPatientNames, use_multiprocessing):

    curOutputPath = f'{outputPath}/L3L4CenterCoordinates.xlsx'  
    curOutputPathOrgSpacing = f'{outputPath}/L3L4CenterCoordinates_InOriginalSpacing.xlsx'

    if use_multiprocessing:
        resultCenters = [i.get()[0] for i in results]
        resultCenters_OrgSpacing = [i.get()[1] for i in results]
    else:
        resultCenters = [results[0]]
        resultCenters_OrgSpacing = [results[1]]
    
    if os.path.isfile(curOutputPath):
        table = pd.read_excel(curOutputPath)
        existingResultCenters = []
        existingResultCenters.extend(pd.DataFrame(table,columns=['Patient_Name','x_gt','y_gt','z_gt','x_pred','y_pred','z_pred','euclid_dist','diff_z','diff_z_in_mm','exist_warning']).values.tolist())
        existingResultCenters = [[existingResultCenters[i][0],existingResultCenters[i][1:4],existingResultCenters[i][4:7],existingResultCenters[i][7],existingResultCenters[i][8],existingResultCenters[i][9],existingResultCenters[i][10]] for i in range(0,len(existingResultCenters))]
        resultCenters += existingResultCenters
    if os.path.isfile(curOutputPathOrgSpacing):
        table = pd.read_excel(curOutputPathOrgSpacing)
        existingResultCenters = []
        existingResultCenters.extend(pd.DataFrame(table,columns=['Patient_Name','x_gt','y_gt','z_gt','x_pred','y_pred','z_pred','euclid_dist','diff_z','diff_z_in_mm','exist_warning']).values.tolist())
        existingResultCenters = [[existingResultCenters[i][0],existingResultCenters[i][1:4],existingResultCenters[i][4:7],existingResultCenters[i][7],existingResultCenters[i][8],existingResultCenters[i][9],existingResultCenters[i][10]] for i in range(0,len(existingResultCenters))]
        resultCenters_OrgSpacing += existingResultCenters

    
    df = {'Patient_Name':[resultCenters[i][0] for i in range(0,len(resultCenters))],
          'x_gt':[resultCenters[i][1][0] for i in range(0,len(resultCenters))],
          'y_gt':[resultCenters[i][1][1] for i in range(0,len(resultCenters))],
          'z_gt':[resultCenters[i][1][2] for i in range(0,len(resultCenters))],
          'x_pred':[resultCenters[i][2][0] for i in range(0,len(resultCenters))],
          'y_pred':[resultCenters[i][2][1] for i in range(0,len(resultCenters))],
          'z_pred':[resultCenters[i][2][2] for i in range(0,len(resultCenters))],
          'euclid_dist':[resultCenters[i][3] for i in range(0,len(resultCenters))],
          'diff_z':[resultCenters[i][4] for i in range(0,len(resultCenters))],
          'diff_z_in_mm':[resultCenters[i][5] for i in range(0,len(resultCenters))],
          'exist_warning':[resultCenters[i][6] for i in range(0,len(resultCenters))]}
    
    df = pd.DataFrame.from_dict(df)
    df.to_csv(curOutputPath.replace('xlsx','csv'))
    df.to_excel(curOutputPath)
    
    df_orgSpacing = {'Patient_Name':[resultCenters_OrgSpacing[i][0] for i in range(0,len(resultCenters_OrgSpacing))],
                     'x_gt':[resultCenters_OrgSpacing[i][1][0] for i in range(0,len(resultCenters_OrgSpacing))],
                     'y_gt':[resultCenters_OrgSpacing[i][1][1] for i in range(0,len(resultCenters_OrgSpacing))],
                     'z_gt':[resultCenters_OrgSpacing[i][1][2] for i in range(0,len(resultCenters_OrgSpacing))],
                     'x_pred':[resultCenters_OrgSpacing[i][2][0] for i in range(0,len(resultCenters_OrgSpacing))],
                     'y_pred':[resultCenters_OrgSpacing[i][2][1] for i in range(0,len(resultCenters_OrgSpacing))],
                     'z_pred':[resultCenters_OrgSpacing[i][2][2] for i in range(0,len(resultCenters_OrgSpacing))],
                     'euclid_dist':[resultCenters_OrgSpacing[i][3] for i in range(0,len(resultCenters_OrgSpacing))],
                     'diff_z':[resultCenters_OrgSpacing[i][4] for i in range(0,len(resultCenters_OrgSpacing))],
                     'diff_z_in_mm':[resultCenters_OrgSpacing[i][5] for i in range(0,len(resultCenters_OrgSpacing))],
                     'exist_warning':[resultCenters_OrgSpacing[i][6] for i in range(0,len(resultCenters_OrgSpacing))]}
    df_orgSpacing = pd.DataFrame.from_dict(df_orgSpacing)
    df_orgSpacing.to_csv(curOutputPathOrgSpacing.replace('xlsx','csv'))
    df_orgSpacing.to_excel(curOutputPathOrgSpacing)

    if saveOriginal:
        # replace nnUNet names with original names
        patNames_nnUNet = df_orgSpacing['Patient_Name'].tolist()
        patNames_nnUNet = [f'{x}_0000.nii.gz' for x in patNames_nnUNet]
        
        indexOrder = []
        for curPatName in patNames_nnUNet:
            indexOrder += [x for x in range(len(newPatientNames)) if newPatientNames[x].endswith(curPatName)]
            
        patNames_Org = [orgPatientNames[x] for x in indexOrder]
        patNames_Org = [os.path.basename(x).split('.')[0] for x in patNames_Org]
        df_orgSpacing['Patient_Name'] = patNames_Org
        
        orgOutputPath = os.path.join(os.sep.join(os.path.dirname(orgPatientNames[0]).split(os.sep)[0:-2]),'nnUNet_L3L4Slice_Results','L3L4CenterCoordinates_InOriginalSpacing.xlsx')
        df_orgSpacing.to_csv(orgOutputPath.replace('xlsx','csv'))
        df_orgSpacing.to_excel(orgOutputPath)

def patientsStillNeedToBePredicted(data,outputPath):
    predResultsTable = []
    for root_path, dirs, files in os.walk(outputPath):
        for name in files:
            if 'L3L4CenterCoordinates' in name and (name.endswith(".xls") or name.endswith(".xlsx")) and 'InOriginalSpacing' not in name: 
                predResultsTable.append(os.path.join(root_path,name))
    predResultsTable.sort()
    
    predPatients = [] 
    for predResults in predResultsTable:
        curTable = pd.read_excel(predResults)
        predPatients.extend(pd.DataFrame(curTable,columns=['Patient_Name']).values.tolist())
         
    notPredPatients = [data[i] for i in range(0,len(data)) if [data[i].split('/')[-1].split('.')[0]] not in predPatients] 
    
    print('Number of Patients still need to be predicted: %s' %len(notPredPatients))
    
    return notPredPatients

def readOriginalDataNames(taskName):
    
    orgPatientNames = []
    newPatientNames = []

    orgPatientNames_excelFiles = []
    nnUNet_path = os.path.expandvars('$nnUNet_raw_data_base') 
    rawDataDir = "%s/nnUNet_raw_data/%s" %(nnUNet_path,taskName)
    
    for root_path, dirs, files in os.walk(rawDataDir):
        for name in files:
            if ('originalAndnnUNetPatientNames' in name or 'originalPatientNames' in name or 'oldAndNewPatientNames') and (name.endswith(".xls") or name.endswith(".xlsx")): 
                orgPatientNames_excelFiles.append(os.path.join(root_path,name))

   
    for orgPatientNames_excelFile in orgPatientNames_excelFiles:
        curTable = pd.read_excel(orgPatientNames_excelFile)
        if len(list(curTable.keys()))>2:
            columnOrgPatientName = list(curTable.keys())[1]
            columnNNUNetPatientName = list(curTable.keys())[2]
        else:
            columnOrgPatientName = list(curTable.keys())[0]
            columnNNUNetPatientName = list(curTable.keys())[1]
        orgPatientNames.extend(pd.DataFrame(curTable,columns=[columnOrgPatientName]).values.tolist())
        newPatientNames.extend(pd.DataFrame(curTable,columns=[columnNNUNetPatientName]).values.tolist())
        
    orgPatientNames = [x[0] for x in orgPatientNames]
    newPatientNames = [x[0] for x in newPatientNames]
    
    return orgPatientNames,newPatientNames

def predict_hd(dataPath, outputPath, gtName,visualization=False,saveL3L4Slice=False,saveOriginal=False,cropExtremeties=False, use_multiprocessing=False):
    
    data = []
    for root_path, dirs, files in os.walk(dataPath):
        for name in files:
            if name.endswith(".nii") or name.endswith(".nii.gz") and 'L3L4Slice' not in name:
                data.append(os.path.join(root_path,name))
    data.sort()
    
    indxTask = data[0].find('Task')
    taskName = data[0][indxTask:]
    indxFilesep = [i for i in range(0,len(taskName)) if taskName[i]=='/']
    taskName = taskName[0:indxFilesep[0]]
    
    needToBePredicted = patientsStillNeedToBePredicted(data, outputPath) 
    
    if saveL3L4Slice:
        l3l4SlicePath = f'{outputPath}/L3L4Slice'

        if not os.path.exists(l3l4SlicePath):
            os.makedirs(l3l4SlicePath)
            
    nnUNet_path = os.path.expandvars('$nnUNet_raw_data_base') 
       
    if visualization:
        sagittalPath = f'{nnUNet_path}/Plots/{taskName}/Pred/sagittal' 
        coronalPath = f'{nnUNet_path}/Plots/{taskName}/Pred/coronal' 
        axialPath = f'{nnUNet_path}/Plots/{taskName}/Pred/axial' 
        
        for curPath in [sagittalPath,coronalPath,axialPath]:
            if not os.path.exists(curPath):
                os.makedirs(curPath)
    
    if len(needToBePredicted):
        rawData,groundTruth = getRawDataAndGT(data)
        orgPatientNames, newPatientNames = readOriginalDataNames(taskName)

        if use_multiprocessing:

        
            pool = mp.Pool(processes = 4)
            
            results = [pool.apply_async(predictCase,args=(d,outputPath,rawData,groundTruth,orgPatientNames,newPatientNames,gtName,visualization,saveL3L4Slice,saveOriginal,cropExtremeties)) for d in needToBePredicted]
            pool.close()
            pool.join()

            saveResults(results, outputPath, saveOriginal, orgPatientNames, newPatientNames, use_multiprocessing)

        else:
            for d in needToBePredicted:
                results = predictCase(d,outputPath,rawData,groundTruth,orgPatientNames,newPatientNames,gtName,visualization,saveL3L4Slice,saveOriginal,cropExtremeties) 
                saveResults(results, outputPath, saveOriginal, orgPatientNames, newPatientNames, use_multiprocessing)

    


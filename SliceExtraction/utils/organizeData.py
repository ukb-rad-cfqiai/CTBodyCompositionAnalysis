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

import os 
import sys
parent_directory = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])
sys.path.append(parent_directory)
import numpy as np
from tqdm import tqdm
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
import datetime
import json
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt

from scripts_HeightDetection.getProbabilityMapForHeightDetection import getProbabilityMap
from utils.plotData import plotData

class DataOrganizer:
    def __init__(self,data_path,taskName,trainPercentage,modality, 
               testSubfolderName='',gtNames='',changeVoxelSpacing=False,
               wantedVoxelSpacing=[],randomSplit=True,
               split_path='',visualize=True, description='',
               labels={'0':'background',
                       '1':'foreground'}, 
               lowestHUValue=-400,
               highestHUValue=600):
        # params:
            # data_path: Path where the data is located
            # taskName: Name of the nnUNet Task
            # trainPercentage: Percentage of training data (rest is test). 
            #                  1/5 of training data is retained in each fold for validation (5 fold cross-validation)
            # modality: Specify modality of your dataset (e.g. CT or MRI)
            # testSubfolderName: If test data should be saved in certain subfolder, else all test data are saved to /imagesTs  
            # gtNames: If ground truth exists you can add here the endings of 
            #          the ground truth files. If they are more than on possible 
            #          ending you can add all options into a list.
            # changeVoxelSpacing: Set to true, if you want to resample data to new voxel spacing 
            #                     Unnecessary if you want to predict only, because data is automatically resampled to median voxel spacing by the nnUnet
            # wantedVoxelSpacing: Specify new spacing if you want to resample data
            # randomSplitting: Set to false if you want a specific splliting of the data into train / validation / test 
            # split_path: If not random splitting specify here the path to .json file, 
            #             which defines spliitting into train / validation / test   
            # visualize: Set to False if you don't want automatic created plots 
            # description: Specify description of your dataset for .json file 
            # labels: Specify label dictionary for the nnUNet 
            
        self.data_path = data_path
        self.taskName = taskName
        self.trainPercentage = trainPercentage 
        self.modality = modality
        self.testSubfolderName = testSubfolderName
        self.gtNames = gtNames
        self.changeVoxelSpacing = changeVoxelSpacing
        self.wantedVoxelSpacing = wantedVoxelSpacing
        if not randomSplit:
            if not os.path.exists(split_path):
                print('WARNING! Path to split.json does not exist. Set randomSplit to True.\n')
                randomSplit = True
            if not '.json' in split_path:
                print('WARNING! split_path is not a path to .json file. Set randomSplit to True.\n')
                randomSplit = True
        self.randomSplit = randomSplit
        self.split_path = split_path
        self.visualize = visualize
        self.description = description
        self.labels  = labels
        self.nnUNet_path = os.path.expandvars('$nnUNet_raw_data_base') # get environment variable
        self.lowestHUValue = lowestHUValue
        self.highestHUValue = highestHUValue
        
       
    def organizePaths(self):
        if self.testSubfolderName and not self.testSubfolderName.endswith('/'):
            self.testSubfolderName = '/%s' %self.testSubfolderName
            
        imagesTrPath = f'{self.nnUNet_path}/nnUNet_raw_data/{self.taskName}/imagesTr' 
        imagesTsPath = f'{self.nnUNet_path}/nnUNet_raw_data/{self.taskName}/imagesTs{self.testSubfolderName}'
        labelsTrPath = f'{self.nnUNet_path}/nnUNet_raw_data/{self.taskName}/labelsTr' 
        labelsTsPath = f'{self.nnUNet_path}/nnUNet_raw_data/{self.taskName}/labelsTs{self.testSubfolderName}'
        
        if not os.path.exists(imagesTrPath):
            os.makedirs(imagesTrPath)
        if not os.path.exists(imagesTsPath):
            os.makedirs(imagesTsPath)
        if not os.path.exists(labelsTrPath):
            os.makedirs(labelsTrPath)
        if not os.path.exists(labelsTsPath):
            os.makedirs(labelsTsPath)
        
        return imagesTrPath, imagesTsPath, labelsTrPath, labelsTsPath
            
    def splitIntoTrainTest(self,data):
        
        randomSplit = self.randomSplit
    
        if randomSplit:
            numberOfTrainingCases = round(self.trainPercentage * len(data))
            permuteData = np.random.permutation(len(data))
            trainingIndizes = np.sort(permuteData[0:numberOfTrainingCases])
            testIndizes = np.sort(permuteData[numberOfTrainingCases:len(permuteData)])
        
            train = [data[i] for i in trainingIndizes]     
            test = [data[i] for i in testIndizes] 
            split_dict = []
            
        else:
            with open(self.split_path) as json_file:
                splitFile = json.load(json_file)
                
            train = []
            test = []
            split_dict = []
            for n in range(len(splitFile)-1):
                oD = OrderedDict()
                oD['train'] = []
                oD['val'] = []
                split_dict.append(oD)
            
            testCase = False
        
            for d in data:
                if d.split('/')[-1] in splitFile['test_data']:
                    test.append(d)
                    testCase = True
                else:
                    train.append(d)
                    testCase = False
            
                if not testCase:
                    for n in range(len(splitFile)-1):
                        if d.split('/')[-1] in splitFile['fold_%s_validation' %n]:
                            split_dict[n]['val'].append(d)
                        else:
                            split_dict[n]['train'].append(d)
                              
        return test,train,split_dict

    def resampleDataToSpecificSpacing(self,image_path,interpolation):
        try:
            itk_image = sitk.ReadImage(image_path,sitk.sitkFloat32)
            original_spacing = itk_image.GetSpacing()
            original_size = itk_image.GetSize()
            
            scaleFactors = [self.wantedVoxelSpacing[0]/original_spacing[0],
                            self.wantedVoxelSpacing[1]/original_spacing[1],
                            self.wantedVoxelSpacing[2]/original_spacing[2]]
            
            out_size = [int(np.round(original_size[0] * (original_spacing[0] / self.wantedVoxelSpacing[0]))),
                        int(np.round(original_size[1] * (original_spacing[1] / self.wantedVoxelSpacing[1]))),
                        int(np.round(original_size[2] * (original_spacing[2] / self.wantedVoxelSpacing[2])))]
                
            resample = sitk.ResampleImageFilter()
            resample.SetOutputSpacing(self.wantedVoxelSpacing)
            resample.SetSize(out_size)
            resample.SetOutputDirection(itk_image.GetDirection())
            resample.SetOutputOrigin(itk_image.GetOrigin())
            resample.SetTransform(sitk.Transform())
            resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
            if interpolation == 'BSpline':
                resample.SetInterpolator(sitk.sitkBSpline)
            elif interpolation == 'NearestNeighbor':
                resample.SetInterpolator(sitk.sitkNearestNeighbor)
            else:
                print('WARNING! Param interpolation should be either BSpline oder NearestNeighbor. BSpline interpolation is used instead')
                resample.SetInterpolator(sitk.sitkBSpline)
        
            resampled_img = resample.Execute(itk_image)
            error = False
        except:
            print(f'WARNING! Resamplin does not work for {image_path}. Image was not resampled!!')
            resampled_img = nib.load(image_path).get_fdata()
            scaleFactors = [1,1,1]
            error = True
        
        return resampled_img, scaleFactors, error
    

    def createAndUpdateJSONFile(self):
        
        dataPath = f'{self.nnUNet_path}/nnUNet_raw_data/{self.taskName}'
        data = []
        for root_path, dirs, files in os.walk(dataPath):
                 for name in files:
                     if name.endswith(".nii") or name.endswith(".nii.gz"):
                         data.append(os.path.join(root_path,name))
                         
         
        json_dict = {}
        json_dict['name'] = self.taskName
        json_dict['description'] = self.description
        
        tempData = nib.load(data[0])
        json_dict['tensorImagSize'] = '%sD' %str(len(tempData.shape))  
        json_dict['reference'] = f'nnunet_{self.taskName}'
        json_dict['licence'] = ''
        json_dict['release'] = '0.0'
        json_dict['modality'] = self.modality
        json_dict['labels'] = self.labels 
        
        trainingImages = [i for i in data if 'imagesTr' in i]
        trainingImages.sort()
        trainingLabels = [i for i in data if 'labelsTr' in i]
        trainingLabels.sort()
        
        testImages = [i for i in data if 'imagesTs' in i]
        testImages.sort()
        
        json_dict["numTraining"] = len(trainingImages)
        json_dict["numTest"] = len(testImages)
        
        json_dict["training"] = []
        
        for i in range(0,len(trainingImages)):
            indxLabelsTr = trainingLabels[i].find('labelsTr')
            indxPat = trainingImages[i].find('pat')
            patWithModality = trainingImages[i][indxPat:]
            indxUnderScore = [i for i in range(0,len(patWithModality)) if patWithModality[i]=='_']
            patWithoutModality =patWithModality[0:indxUnderScore[-1]] + ".nii.gz"
            json_dict["training"].append({"image":"./imagesTr/%s" %patWithoutModality,
                                          "label":"./labelsTr/%s" %patWithoutModality})    
        json_dict["test"] = []
        for i in range(0,len(testImages)):
            indxImagesTs = testImages[i].find('imagesTs')
            indxPat = testImages[i].find('pat')
            testFolder = testImages[i][indxImagesTs:indxPat]
            patWithModality = testImages[i][indxPat:]
            indxUnderScore = [i for i in range(0,len(patWithModality)) if patWithModality[i]=='_']
            patWithoutModality =patWithModality[0:indxUnderScore[-1]] + ".nii.gz"
            json_dict["test"].append("./%s%s" %(testFolder,patWithoutModality))
            
        with open(f'{dataPath}/dataset.json', 'w') as outfile:
            json.dump(json_dict, outfile,ensure_ascii=False, indent=4)
       

    def organizeData(self):
        
        data = []
        for root_path, dirs, files in os.walk(self.data_path):
                 for name in files:
                     if (name.endswith(".nii.gz") or name.endswith(".nii")) and 'label' not in name and 'L3L4Slice' not in name and 'segm' not in name and 'roi' not in name.lower():
                         data.append(os.path.join(root_path,name))
                         
        
        imagesTrPath, imagesTsPath, labelsTrPath, labelsTsPath = self.organizePaths()
        test, train, split_dict = self.splitIntoTrainTest(data)
        
        # Does outputDir already contain any data?
        outputDir = f'{self.nnUNet_path}/nnUNet_raw_data/{self.taskName}'
        alredy_existing_data = []
        for root_path, dirs, files in os.walk(outputDir):
                 for name in files:
                     if name.endswith(".nii") or name.endswith(".nii.gz"):
                         alredy_existing_data.append(os.path.join(root_path,name))
        
        if len(alredy_existing_data) != 0:
            trainingSamples = [int(i.split('/')[-1].split('.')[0].split('_')[1]) for i in alredy_existing_data if "imagesTr" in i]
            testSamples = [int(i.split('/')[-1].split('.')[0].split('_')[1]) for i in alredy_existing_data if "imagesTs" in i]
            if not trainingSamples:
                trainingSamples = [0]
            if not testSamples:
                testSamples = [0]
            numExistingPat = max(max(trainingSamples),max(testSamples))
        else:
            numExistingPat = 0
        
        if self.visualize:
            plotPath = f'{self.nnUNet_path}/Plots/{self.taskName}' 
            sagittalPath = f'{plotPath}/GroundTruth/sagittal'
            coronalPath = f'{plotPath}/GroundTruth/coronal'
            axialPath = f'{plotPath}/GroundTruth/axial'
            
            if not os.path.exists(sagittalPath):
                os.makedirs(sagittalPath)
            if not os.path.exists(coronalPath):
                os.makedirs(coronalPath)
            if not os.path.exists(axialPath):
                os.makedirs(axialPath)
       
        data_path_dict = {'original_path':[],
                          'nnUNet_path':[]}
       
        trainSplit = []
        valSplit = []
        for n in range(len(split_dict)):
            trainSplit.append([])
            valSplit.append([])
        
        print('---------- START TO ORGANIZE DATA ----------')
        for id_cur_image in tqdm(range(0,len(data))):
            
            # load current image
            cur_data = nib.load(data[id_cur_image])
            cur_image = cur_data.get_fdata()
            cur_filename = cur_data.get_filename()
            
            searchGTPath = os.path.dirname(cur_filename)
            
            if self.gtNames == '':
                existGT = False
                cur_label = None
            else:
                existGT = True
    
                cur_gt_path = []
                for root_path, dirs, files in os.walk(searchGTPath):
                     for name in files:
                         if self.gtNames in name:
                             cur_gt_path.append(os.path.join(root_path,name))
                if cur_gt_path:
                    if len(cur_gt_path) > 1:
                        print(f'Warning! Found more than one ground truth for current data {cur_data.get_filename()}. Used the first')
                    cur_gt_path = cur_gt_path[0]
                else:
                    print(f'Could not find ground truth for {cur_data.get_filename()}')
                    existGT = False
                    cur_label = None
            
            coordGT = False
            segmGT = False
            coord = None
            if existGT:
                if '.txt' in cur_gt_path:
                    coordGT = True
                    coord = open(cur_gt_path)
                    coord = coord.read().split(',')
                    coord = [int(coord[0]),int(coord[1]),int(coord[2])]
                elif '.nii' or '.nii.gz' in cur_gt_path:
                   segmGT = True
                   cur_label = nib.load(cur_gt_path)
                else:
                    raise('ERROR! Ground Truth has to be either a .txt file in the form x_coord,y_coord,z_coord for height detection or a Nifti file for segmentation tasks')
                
            # create patient Name
            i_str = str(id_cur_image+numExistingPat+1)
            neededDigits = 6-len(i_str) 
            for d in range(0,neededDigits):
                i_str = '0' + i_str
            pat_Name = f'pat_{i_str}_0000' 
            
            if cur_filename in train:
                outputDirImage = f'{imagesTrPath}/{pat_Name}.nii.gz' 
                indxUnderscore = [x for x in range(0,len(pat_Name)) if pat_Name[x]=='_']
                outputDirLabel = f'{labelsTrPath}/{pat_Name[0:indxUnderscore[1]]}.nii.gz' 
            elif cur_filename in test:
                outputDirImage = f'{imagesTsPath}/{pat_Name}.nii.gz' 
                indxUnderscore = [x for x in range(0,len(pat_Name)) if pat_Name[x]=='_']
                outputDirLabel = f'{labelsTsPath}/{pat_Name[0:indxUnderscore[1]]}.nii.gz'
            
            if coordGT:
                # write coordinate to text file
                indxDot = outputDirImage.find('.')
                txt_path = f'{outputDirImage[0:indxDot]}_L3L4Coordinates.txt' 
                f = open(txt_path,"a")
                f.write(str(coord[0])+','+str(coord[1])+','+str(coord[2]))
                f.close()
                
            # voxel spacing
            if self.changeVoxelSpacing:
                resampled_img, scaleFactors, error = self.resampleDataToSpecificSpacing(cur_filename,interpolation='BSpline')
                
                if not error:
     
                    writer = sitk.ImageFileWriter();
                    writer.SetFileName(outputDirImage)
                    writer.Execute(resampled_img)
                    
                    # NOW SAVE AGAIN WITH CORRECT HEADER
                    new_data = nib.load(outputDirImage)
                    if cur_data.header.extensions:  
                        new_data.header.extensions =cur_data.header.extensions
                        
                    cur_data = new_data
                    cur_image = cur_data.get_fdata()
                    
                    if coordGT:
                        coord_resized = [round(coord[0]/scaleFactors[0]),round(coord[1]/scaleFactors[1]),round(coord[2]/scaleFactors[2])]
                        
                        # write coordinate to text file
                        indxDot = outputDirImage.find('.')
                        txt_path = f'{outputDirImage[0:indxDot]}_L3L4Coordinates_AfterResizing.txt'
                        f = open(txt_path,"a")
                        f.write('CoordinateOld: ' + str(coord[0])+','+str(coord[1])+','+str(coord[2])+'\n'+
                                'CoordinateNew: ' + str(coord_resized[0])+','+str(coord_resized[1])+','+str(coord_resized[2])+'\n'+
                                'ScalingFactors: ' + str(scaleFactors[0]) + ',' + str(scaleFactors[1]) + ',' + str(scaleFactors[2]))
                        f.close()
                        
                        coord = coord_resized
                    else:
                        indxDot = outputDirImage.find('.')
                        txt_path = '%s_ScalingFactors.txt' %outputDirImage[0:indxDot]
                        f = open(txt_path,"a")
                        f.write('ScalingFactors: ' + str(scaleFactors[0]) + ',' + str(scaleFactors[1]) + ',' + str(scaleFactors[2]))
                        f.close()
                    
                    if segmGT:
                        resampled_labels,_ = self.resampleDataToSpecificSpacing(cur_gt_path,interpolation='NearestNeighbour')
                       
                        writer = sitk.ImageFileWriter();
                        writer.SetFileName(outputDirLabel)
                        writer.Execute(resampled_labels)
                        
                        # NOW SAVE AGAIN WITH CORRECT HEADER
                        new_label = nib.load(outputDirLabel)
                        if cur_label.header.extensions:  
                            new_label.header.extensions =cur_label.header.extensions
                            
                        cur_label = new_label
                else:
                    with open(outputDirImage.replace('.nii.gz','_warning_Resampling.txt'),'w') as f:
                        f.writelines('Error in resampling data. Data was not resampled. Continue with next data')
                        f.close()
                    continue
                    
            # write data   
            nib.save(cur_data,outputDirImage)
            
            if coordGT: 
                # write label
                probMap = getProbabilityMap(cur_image,coord)
                cur_label = nib.Nifti1Image(probMap, cur_data.affine)
                
            if existGT:
                nib.save(cur_label,outputDirLabel)
                
            if self.visualize:
                if existGT:
                    resolutionsRatioZ = cur_data.header.get_zooms()[2]/cur_data.header.get_zooms()[1]
                    fig_axial = plotData(image=cur_image,coord_gt=coord,label=cur_label.get_fdata(),
                                         lowestValue=self.lowestHUValue,highestValue=self.highestHUValue,plane='axial',
                                         resolutionsRatioZ=resolutionsRatioZ)
                    fig_sagittal = plotData(image=cur_image,coord_gt=coord,label=cur_label.get_fdata(),
                                        lowestValue=self.lowestHUValue,highestValue=self.highestHUValue,plane='sagittal',
                                        resolutionsRatioZ=resolutionsRatioZ)
                    fig_coronal = plotData(image=cur_image,coord_gt=coord,label=cur_label.get_fdata(),
                                       lowestValue=self.lowestHUValue,highestValue=self.highestHUValue,plane='coronal',
                                       resolutionsRatioZ=resolutionsRatioZ)
                
                    fig_axial.savefig(f'{axialPath}/{pat_Name}_axial.png',dpi=300) 
                    fig_sagittal.savefig(f'{sagittalPath}/{pat_Name}_sagittal.png',dpi=300)
                    fig_coronal.savefig(f'{coronalPath}/{pat_Name}_coronal.png',dpi=300)
                    plt.close(fig_axial)
                    plt.close(fig_sagittal)
                    plt.close(fig_coronal)
                else:
                    resolutionsRatioZ = cur_data.header.get_zooms()[2]/cur_data.header.get_zooms()[1]
                    fig_axial = plotData(image=cur_image,
                                         lowestValue=self.lowestHUValue,highestValue=self.highestHUValue,plane='axial',
                                         resolutionsRatioZ=resolutionsRatioZ)
                    fig_sagittal = plotData(image=cur_image,
                                        lowestValue=self.lowestHUValue,highestValue=self.highestHUValue,plane='sagittal',
                                        resolutionsRatioZ=resolutionsRatioZ)
                    fig_coronal = plotData(image=cur_image,
                                       lowestValue=self.lowestHUValue,highestValue=self.highestHUValue,plane='coronal',
                                       resolutionsRatioZ=resolutionsRatioZ)
                
                    fig_axial.savefig(f'{axialPath}/{pat_Name}_axial.png',dpi=300) 
                    fig_sagittal.savefig(f'{sagittalPath}/{pat_Name}_sagittal.png',dpi=300)
                    fig_coronal.savefig(f'{coronalPath}/{pat_Name}_coronal.png',dpi=300)
                    plt.close(fig_axial)
                    plt.close(fig_sagittal)
                    plt.close(fig_coronal)
       
            if not self.randomSplit or len(split_dict)>0:          
                for n in range(len(split_dict)):
                    if cur_filename in split_dict[n]['train']:
                        trainSplit[n].append('_'.join(pat_Name.split('_')[0:2]))
                    elif cur_filename in split_dict[n]['val']:
                        valSplit[n].append('_'.join(pat_Name.split('_')[0:2]))
                
               
            data_path_dict['original_path'].append(data[id_cur_image])
            data_path_dict['nnUNet_path'].append(outputDirImage)
    
        # write to Excel File
        df = pd.DataFrame(data_path_dict)
        now = datetime.datetime.now()
        csvPath = f'{self.nnUNet_path}/nnUNet_raw_data/{self.taskName}/originalAndnnUNetPatientNames_{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv' 

        df.to_csv(csvPath)
        df.to_excel(csvPath.replace('.csv','.xlsx'))
        
        if not self.randomSplit:
            final_split_dict = []
            for n in range(len(split_dict)):
                oD = OrderedDict()
                oD['train'] = np.array(trainSplit[n])
                oD['val'] = np.array(valSplit[n])
                final_split_dict.append(oD)
            
            nnUNet_preprocessed_path = os.path.expandvars('$nnUNet_preprocessed') # get environment variable
            output_path = f'{nnUNet_preprocessed_path}/{self.taskName}' 
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(f'{output_path}/splits_final.pkl', 'wb') as handle:
                pickle.dump(final_split_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                                
        self.createAndUpdateJSONFile()
        print('---------- FINISHED ORGANIZE DATA ----------')

                  
                      
                      

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
import argparse
from utils.organizeData import *
from scripts_HeightDetection.predict_heightDetection import predict_hd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help = 'data path', required=True, type=str)
    parser.add_argument('--model_path', help='Path where trained model is saved',required=True,type=str)
    parser.add_argument('--modality', help='Modality of your dataset (e.g. CT or MRI)', required=False,type=str,default='CT')
    parser.add_argument('--test_subfolder_name', help='If you want to specify certain subfolder for your test data. Default test data is saved to /imagesTs', required=False, type=str, default='')
    parser.add_argument('--gtNames',help='If you have GT specify here end of ground truth filenames',nargs='?',const='', default="", required=False)
    parser.add_argument('--description',help='If you want to add a description of your dataset to the .json file for the nnUNet',required=False,type=str,default='')
    parser.add_argument('--num_classes',help='Specify here the number of classes / labels',required=False,type=int,default=1)  
    parser.add_argument('--applyQC',help='Set if you want to apply quality control',required=False,action="store_true")
    parser.add_argument('--use_multiprocessing',help='Set if you want to apply multiprocessing during post-processing',required=False,action="store_true")
    parser.add_argument('--visualize',help='Set if you want to generate plots of your data',required=False,action="store_true")
    parser.add_argument('--qcModelPath',nargs='?',const='', default="", required=False,
                      help='Path to model.pkl for quality control (if you want to apply QC, set applyQC to True')
    parser.add_argument('--model',help='Which model do you want to use for prediction (final checkpoint or model_best?', required=False,default='model_final_checkpoint')
    parser.add_argument('--changeSpacing',help='Set to true if you want to change voxel spacing',required=False,action="store_true")
    parser.add_argument('--wantedSpacing', nargs='*', help='If you want to change voxel spacing, specify here wanted spacing', required=False,type=float,default=[])
    args = parser.parse_args()
    
    data_path = args.d
    model_path = args.model_path
    modality = args.modality
    testSubfolderName = args.test_subfolder_name
    gtNames = args.gtNames
    taskName = model_path[model_path.find('Task'):].split(os.sep)[0]
    trainPercentage=0 # prediction
    visualize = args.visualize
    description = args.description
    num_classes = args.num_classes
    applyQC = args.applyQC
    qcModelPath = args.qcModelPath
    model=args.model
    changeSpacing = args.changeSpacing
    wantedSpacing=args.wantedSpacing
    use_multiprocessing=args.use_multiprocessing
    
    label_dict = {}
    for cur_label in range(num_classes+1):
        if cur_label == 0:
            label_dict[cur_label] = 'background'
        else:
            label_dict[cur_label] = f'class_{str(cur_label)}'
    
    if args.modality=='CT':
        lowestHUValue = -400
        highestHUValue = 600
    else:
        lowestHUValue = None
        highestHUValue = None
        
   
    dataOrganizer = DataOrganizer(data_path,taskName,trainPercentage,modality, 
                                  testSubfolderName,gtNames,
                                  visualize=visualize,description=description,
                                  labels=label_dict, changeVoxelSpacing=changeSpacing,
                                  wantedVoxelSpacing=wantedSpacing,
                                  lowestHUValue=lowestHUValue, highestHUValue=highestHUValue) 

    dataOrganizer.organizeData()
    
    # start nnUNet prediction
    config = model_path[model_path.find('nnUNet_trained_models'):].split('/')[2]
    taskID = taskName.split('_')[0].replace('Task','')
    trainer = model_path.split('/')[-1].replace('__nnUNetPlansv2.1','')
    inputPath = f'{os.path.expandvars("$nnUNet_raw_data_base")}/nnUNet_raw_data/{taskName}/imagesTs/{testSubfolderName}'
    outputPath = f'{"/".join(os.path.expandvars("$nnUNet_raw_data_base").split("/")[0:-1])}/nnUNet_predictions/{config}/{taskName}/{trainer}_{model}/{testSubfolderName}'
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    
    if applyQC:
        os.system(f'nnUNet_predict -chk {model} -i {inputPath} -o {outputPath} -t {taskID} -m {config} -tr {trainer} --applyQC --qcModelPath {qcModelPath} --num_threads_preprocessing 2 --save_npz')
    else:
        os.system(f'nnUNet_predict -chk {model} -i {inputPath} -o {outputPath} -t {taskID} -m {config} -tr {trainer} --num_threads_preprocessing 2 --save_npz')
    
   
    predict_hd(outputPath,outputPath,gtNames,visualize,True,True,use_multiprocessing)
    
if __name__ == "__main__":
    main()

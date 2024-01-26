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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help = 'data path', required=True, type=str)
    parser.add_argument('--taskName', help='task name of the model, e.g. Task100_SliceExtraction',required=True,type=str)
    parser.add_argument('--modality', help='Modality of your dataset (e.g. CT or MRI)', required=False,type=str,default='CT')
    parser.add_argument('--test_subfolder_name', help='If you want to specify certain subfolder for your test data. Default test data is saved to /imagesTs', required=False, type=str, default='')
    parser.add_argument('--description',help='If you want to add a description of your dataset to the .json file for the nnUNet',required=False,type=str,default='')
    parser.add_argument('--num_classes',help='Specify here the number of classes / labels',required=False,type=int,default=1)  
    parser.add_argument('--visualize',help='Set if you want to generate plots of your data',required=False,action="store_true")
    parser.add_argument('--model',help='Which model do you want to use for prediction (final checkpoint or model_best?', required=False,default='model_final_checkpoint')
    parser.add_argument('--changeSpacing',help='Set to true if you want to change voxel spacing',required=False,action="store_true")
    parser.add_argument('--wantedSpacing', nargs='*', help='If you want to change voxel spacing, specify here wanted spacing', required=False,type=float,default=[])
    parser.add_argument('--trainPercentage',help='Specify here the percentage of training cases (e.g. 0.8 for 80% training cases)',required=False,type=float,default=0.8)  

    args = parser.parse_args()
    
    data_path = args.d
    modality = args.modality
    testSubfolderName = args.test_subfolder_name
    gtNames = "_L3L4Coordinates.txt"
    taskName = args.taskName
    trainPercentage=args.trainPercentage
    visualize = args.visualize
    description = args.description
    num_classes = args.num_classes
    changeSpacing = args.changeSpacing
    wantedSpacing=args.wantedSpacing
    
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
    
if __name__ == "__main__":
    main()

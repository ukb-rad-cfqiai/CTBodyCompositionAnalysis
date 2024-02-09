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

import sys, os, pathlib, cv2, logging, argparse, pydicom
from tqdm import tqdm
import nibabel as nib
from skimage.transform import resize
from fastai.vision.all import *
from utils.loader_utils import (make_fp16, get_y_by_labelsFolder,
                         TensorNifti, PILNifti)
from utils.image_utils import (write_nii, imfill, getLargestContour, getHalfXandThirdYOfBody, 
                         printImageLabelOverlay, printImageLabelOverlay_FMF, 
                         createMuscleQualityLabel, createdWindowedAndErodedVisceralMasks)

logger = logging.getLogger(__name__)
SEGM_CLASSES= ['bg','muscle','vat','sat']
CLASS_MUSCLE = 1
CLASS_VISCERAL_FAT = 2
CLASS_SUBCUTAN_FAT = 3
CLASS_MUSCLE_FAT = 4
CLASS_MUSCLE_LOW = 5
WINDOW_MUSCLE_LOW = [ -29,  29  ]
WINDOW_MUSCLE_HIGH  = [ 29,  100  ] 
WINDOW_FAT = [ -190 , -30  ]
TARGET_IMSIZE = (512, 512)
TARGET_IMRANGE = (-400, 600)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputPath', type=str, required=False, default=None, help='Full path to input (default ./data)')
    parser.add_argument('--dataType', type=str, required=False, default='.nii.gz', help='DataType of images and masks (default .nii.gz)')
    parser.add_argument('--batch-size', type=int, required=False, default=1, help='Batch size with with inference is done.')
    parser.add_argument('--num-workers', type=int, required=False, default=4, help='How many workers/threads for dataloading.')
    parser.add_argument('--faster', action='store_true', help='No crossval ensemble so faster')
    parser.add_argument('--cpu', action='store_true', help='Run on CPU')

    args = parser.parse_args()
    args.basePath = os.path.dirname(os.path.realpath(__file__))+os.sep
    if args.inputPath is None: args.inputPath = os.path.join(args.basePath, 'data')
    args.device = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
    
    return args

def main():

    args = parse_args()
    logging.root.handlers = []
    handlers=[logging.StreamHandler(sys.stdout)]
    handlers.append( logging.FileHandler(os.path.join(args.inputPath,'bodycomp_inference.log')))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers)
    logging.info(args)
    
    if args.faster: segmentationSplits = [0]
    else: segmentationSplits = [0, 1, 2, 3, 4]
    modelsPath = os.path.join( args.basePath, 'models') 
    files = [f for f in os.listdir(args.inputPath) if not '.' in f and os.path.isfile(os.path.join(args.inputPath, f))]
    
    for f in files:
        try: 
            logging.info(f'{f}: try to convert into nifti.')
            ds = pydicom.dcmread((os.path.join(args.inputPath,f)))
        except:
            logging.info('not a dicom file ... skip converting.')
            continue
        
        b = ds.RescaleIntercept
        m = ds.RescaleSlope
        image = m * ds.pixel_array + b
        image = np.flip( np.swapaxes(np.asarray(image),0,-1), 1)
        image[image>600] = 600; image[image<-400] = -400
        
        curVoxelspacing = [ float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.SliceThickness)   ]
            
        #create affine
        u = np.asarray(ds.ImageOrientationPatient[0:3]) 
        v = np.asarray(ds.ImageOrientationPatient[3:6]) 
        w = np.cross(u, v) 
        u *= curVoxelspacing[0]
        v *= curVoxelspacing[1]
        w *= curVoxelspacing[2]
        affine = np.eye(4)
        affine[0:3,0] = u
        affine[0:3,1] = v
        affine[0:3,2] = w
        affine[0:3,3] = np.asarray(ds.ImagePositionPatient[0:3])
        write_nii(image, affine, curVoxelspacing, os.path.join(args.inputPath,f)+args.dataType)
        logging.info(f'Done converting ...')

    nii_paths = []
    for path in pathlib.Path(args.inputPath).rglob('*'+args.dataType):
        path = str(path)
        if not ('_segm'+args.dataType in path):
            nii_paths.append( str(path) )
    nii_names = [os.path.basename(x) for x in nii_paths]
    
    #now load niis, get body mask crop to body mask and make 512x512
    logging.info(f'preProc: Cropping to body mask ...')
    for path in nii_paths:   
        niiFile = nib.load(path)
        im = np.squeeze(niiFile.get_fdata())
        im[im>TARGET_IMRANGE[1]] = TARGET_IMRANGE[1]; im[im<TARGET_IMRANGE[0]] = TARGET_IMRANGE[0]
        imsize = im.shape
        header = niiFile.header
        affine = niiFile.affine
        voxelspacing = np.array(header.get_zooms())

        mask = getLargestContour(cv2.morphologyEx(np.uint8(im>-400), cv2.MORPH_OPEN, np.uint8(np.ones((7,7)))))>0
        indicies_mask = np.ix_(mask.any(1), mask.any(0))
        if len(list(indicies_mask[0])) > 1 and len(indicies_mask[1][0]) > 1:
            crop_borders = [ indicies_mask[0].min()-10, indicies_mask[0].max()+10, indicies_mask[1].min()-10, indicies_mask[1].max()+10,]
            crop_borders = [0 if x<0 else x for x in crop_borders ]
            len_border_x = crop_borders[3]-crop_borders[2] 
            len_border_y = crop_borders[1]-crop_borders[0]
            
            if len_border_x<480 and len_border_y<480:
                diff_len_borders = np.abs(len_border_x - len_border_y)
                delta1 = int(diff_len_borders/2)
                delta2 = int(diff_len_borders/2) 
                if np.mod(diff_len_borders, 2) != 0: delta2 += 1
        
                if len_border_y > len_border_x:
                    crop_borders[2] -= delta1
                    crop_borders[3] += delta2
                    if crop_borders[2]<0:
                        crop_borders[3] -= crop_borders[2]
                        crop_borders[2] = 0
                    elif crop_borders[3]>imsize[1]:
                        crop_borders[2] -= crop_borders[3]-imsize[1]
                        crop_borders[3] = imsize[1]     
                elif len_border_y > len_border_x:     
                    crop_borders[0] -= delta1
                    crop_borders[1] += delta2
                    if crop_borders[0]<0:
                        crop_borders[1] -= crop_borders[0]
                        crop_borders[0] = 0
                    elif crop_borders[1]>imsize[1]:
                        crop_borders[0] -= crop_borders[1]-imsize[0]
                        crop_borders[1] = imsize[0]
                    
                im = im[crop_borders[0]:crop_borders[1], crop_borders[2]:crop_borders[3]]
            
        else:
            logging.info(f'WARNING: {path} has empty image ...')
            
        cur_imsize = im.shape
        if np.any( np.asarray(cur_imsize) != np.asarray(TARGET_IMSIZE)):
            def resize_and_pad_image(im, target_imsize=(512, 512), padding_mode='constant', cval=-400):
                y_scale = target_imsize[0] / im.shape[0]
                x_scale = target_imsize[1] / im.shape[1]
                scale_factor = min(y_scale, x_scale)
                new_size = (int(im.shape[0] * scale_factor), int(im.shape[1] * scale_factor))
                resized_im = resize(im, new_size, order=3, anti_aliasing=True, mode='constant', cval=cval)
                
                pad_y = (target_imsize[0] - resized_im.shape[0]) // 2
                pad_x = (target_imsize[1] - resized_im.shape[1]) // 2
                padded_im = np.pad(
                    resized_im,
                    ((pad_y, target_imsize[0] - resized_im.shape[0] - pad_y),
                     (pad_x, target_imsize[1] - resized_im.shape[1] - pad_x)), 
                    mode=padding_mode,
                    constant_values=cval)
                return padded_im
            im = resize_and_pad_image(im, TARGET_IMSIZE, cval=TARGET_IMRANGE[0])
            voxelspacing[0] /= im.shape[0] / cur_imsize[0]; voxelspacing[1] /= im.shape[1] / cur_imsize[1]; 
            pixelspacing = voxelspacing[:2]
            affine = nib.affines.rescale_affine(affine, cur_imsize+(1,), tuple(pixelspacing)+(affine[2][2],), im.shape+(1,))
             
        write_nii(im, affine, voxelspacing, path)
    
    #implants detection for QC
    QC_implants_det = []
    for path in nii_paths:   
        cropped_im =  cv2.medianBlur( np.float32(getHalfXandThirdYOfBody(nib.load(path).get_fdata())), 5)
        if  np.max(cropped_im[:]) >= 3000: QC_implants_det.append(1)
        else: QC_implants_det.append(0)

    learn_inference = load_learner(os.path.join(modelsPath,'CDFNet_bestModel_0.pkl'), cpu=args.device=='cpu').to_fp16()
    test_dl = learn_inference.dls.test_dl(nii_paths, bs=args.batch_size, num_workers=args.num_workers, 
                                          with_labels=False, shuffle=False, drop_last=False)                       
    preds = None
    logging.info(f'Predicting on {args.device}')
    for split in segmentationSplits:
        if not args.faster: logging.info(f'Predicting split {split} on {args.device}')
        learn_inference = load_learner(os.path.join(modelsPath,f'CDFNet_bestModel_{split}.pkl'), cpu=args.device=='cpu').to_fp16()
        curPreds, _  = learn_inference.get_preds(dl=test_dl) 
        curPreds = curPreds.cpu().numpy()
        if preds is None: preds = curPreds
        else:  preds += curPreds
    preds /= len(segmentationSplits)
    masks = np.argmax(preds, axis=1)
    
    #get muscle entropy for QC
    QC_entropy_muscle = []
    QC_pred_dice_muscle = []
    for i in range(len(preds)):
        cur_muscle_mask = (masks[i] == 1).flatten()
        cur_muscle_mask_not = np.logical_not(cur_muscle_mask)
        cur_pred = preds[i,1,:].flatten()
        #https://arxiv.org/pdf/1911.13273.pdf PROBLEM: ln of prob not being class np.log(1 - curPred) can be ln(0) ! 
        cur_pred[cur_pred>0.9999] = 0.9999
        cur_pred[cur_pred<0.0001] = 0.0001
        entropy = -1*(cur_pred * np.log(cur_pred) + (1 - cur_pred) * np.log(1 - cur_pred))
        QC_entropy_muscle.append( np.mean(entropy[cur_muscle_mask]) ) 
        QC_pred_dice_muscle.append(  1.016 - QC_entropy_muscle[-1]*0.688 ) 
        if QC_pred_dice_muscle[-1] < 0:  QC_pred_dice_muscle[-1] = 0
        elif QC_pred_dice_muscle[-1] > 1:  QC_pred_dice_muscle[-1] = 1                          
    QC_pred_dice_muscle_below_0_9 = [ int(x<0.9) for x in QC_pred_dice_muscle] 
    
    numVoxel_total = np.array(masks[0]).size
    PixelSpacing1 = np.empty(len(nii_paths)); PixelSpacing1[:] = np.NaN
    PixelSpacing2 = np.empty(len(nii_paths)); PixelSpacing2[:] = np.NaN
    
    totalMuscleInclFat_numVoxel = np.empty(len(nii_paths)); totalMuscleInclFat_numVoxel[:] = np.NaN
    totalMuscleExclFat_numVoxel = np.empty(len(nii_paths)); totalMuscleExclFat_numVoxel[:] = np.NaN
    fatTissueInMuscle_numVoxel = np.empty(len(nii_paths)); fatTissueInMuscle_numVoxel[:] = np.NaN
    muscleLowHu_numVoxel = np.empty(len(nii_paths)); muscleLowHu_numVoxel[:] = np.NaN
    muscleHighHu_numVoxel = np.empty(len(nii_paths)); muscleHighHu_numVoxel[:] = np.NaN
    SAT_numVoxel = np.empty(len(nii_paths)); SAT_numVoxel[:] = np.NaN
    VAT_numVoxel = np.empty(len(nii_paths)); VAT_numVoxel[:] = np.NaN
    VAT_win_numVoxel = np.empty(len(nii_paths)); VAT_win_numVoxel[:] = np.NaN
    VAT_win_erod_numVoxel = np.empty(len(nii_paths)); VAT_win_erod_numVoxel[:] = np.NaN
    
    totalMuscleInclFat_AreaInMM = np.empty(len(nii_paths)); totalMuscleInclFat_AreaInMM[:] = np.NaN
    totalMuscleExclFat_AreaInMM = np.empty(len(nii_paths)); totalMuscleExclFat_AreaInMM[:] = np.NaN
    fatTissueInMuscle_AreaInMM = np.empty(len(nii_paths)); fatTissueInMuscle_AreaInMM[:] = np.NaN
    muscleLowHu_AreaInMM = np.empty(len(nii_paths)); muscleLowHu_AreaInMM[:] = np.NaN
    muscleHighHu_AreaInMM = np.empty(len(nii_paths)); muscleHighHu_AreaInMM[:] = np.NaN
    SAT_AreaInMM = np.empty(len(nii_paths)); SAT_AreaInMM[:] = np.NaN
    VAT_AreaInMM = np.empty(len(nii_paths)); VAT_AreaInMM[:] = np.NaN
    VAT_win_AreaInMM = np.empty(len(nii_paths)); VAT_win_AreaInMM[:] = np.NaN
    VAT_win_erod_AreaInMM = np.empty(len(nii_paths)); VAT_win_erod_AreaInMM[:] = np.NaN
    
    totalMuscleInclFat_mean = np.empty(len(nii_paths)); totalMuscleInclFat_mean[:] = np.NaN
    totalMuscleInclFat_std = np.empty(len(nii_paths)); totalMuscleInclFat_std[:] = np.NaN
    totalMuscleExclFat_mean = np.empty(len(nii_paths)); totalMuscleExclFat_mean[:] = np.NaN
    totalMuscleExclFat_std = np.empty(len(nii_paths)); totalMuscleExclFat_std[:] = np.NaN
    fatTissueInMuscle_mean = np.empty(len(nii_paths)); fatTissueInMuscle_mean[:] = np.NaN
    fatTissueInMuscle_std = np.empty(len(nii_paths)); fatTissueInMuscle_std[:] = np.NaN
    muscleLowHu_mean = np.empty(len(nii_paths)); muscleLowHu_mean[:] = np.NaN
    muscleLowHu_std = np.empty(len(nii_paths)); muscleLowHu_std[:] = np.NaN
    muscleHighHu_mean = np.empty(len(nii_paths)); muscleHighHu_mean[:] = np.NaN
    muscleHighHu_std = np.empty(len(nii_paths)); muscleHighHu_std[:] = np.NaN
    SAT_mean = np.empty(len(nii_paths)); SAT_mean[:] = np.NaN
    SAT_std = np.empty(len(nii_paths)); SAT_std[:] = np.NaN
    VAT_mean = np.empty(len(nii_paths)); VAT_mean[:] = np.NaN
    VAT_std = np.empty(len(nii_paths)); VAT_std[:] = np.NaN
    VAT_win_mean = np.empty(len(nii_paths)); VAT_win_mean[:] = np.NaN
    VAT_win_std = np.empty(len(nii_paths)); VAT_win_std[:] = np.NaN
    VAT_win_erod_mean = np.empty(len(nii_paths)); VAT_win_erod_mean[:] = np.NaN
    VAT_win_erod_std = np.empty(len(nii_paths)); VAT_win_erod_std[:] = np.NaN
    
    FatFreeMuscleFraction = np.empty(len(nii_paths)); FatFreeMuscleFraction[:] = np.NaN
    FattyMuscleFraction = np.empty(len(nii_paths)); FattyMuscleFraction[:] = np.NaN
    IntermuscleFatFraction = np.empty(len(nii_paths)); IntermuscleFatFraction[:] = np.NaN
    
    for idx, path in enumerate(tqdm(nii_paths)):
        
        name = path.split(os.sep)[-1]
        name = name.split('.')[0]
        curDirname = os.path.dirname(path)+os.sep
    
        niiFile = nib.load(str(path))
        im = np.squeeze(niiFile.get_fdata())
        im_shape = np.asarray(im.shape)
        voxelSpacing = niiFile.header.get_zooms()
        
        mask = masks[idx]
        mask_shape = np.asarray(mask.shape)
        if any(im_shape != mask_shape):  mask = cv2.resize(mask, im_shape[::-1], interpolation=cv2.INTER_NEAREST)
        write_nii(mask, niiFile.affine, voxelSpacing, path.replace(args.dataType,'_segm'+args.dataType), as_uint8=True)
 
        PixelSpacing1[idx] = voxelSpacing[0]
        PixelSpacing2[idx] = voxelSpacing[1]
        factorNumToArea = PixelSpacing1[idx] * PixelSpacing2[idx]
       
        muscleQualityLabel = createMuscleQualityLabel(im, mask)
        imgToPrint = np.copy(im)
        imgToPrint[imgToPrint<(-1)*400] = (-1)*400
        imgToPrint[imgToPrint>600] = 600
        printImageLabelOverlay(imgToPrint,muscleQualityLabel, path.replace(args.dataType,'_segm.png'),
                               printImageWithoutOverlay=True, num_classes=len(SEGM_CLASSES))
        
        FMF_label = np.zeros(np.shape(muscleQualityLabel))
        FMF_label[muscleQualityLabel==CLASS_MUSCLE] = 1
        FMF_label[muscleQualityLabel==CLASS_MUSCLE_LOW] = 2
        printImageLabelOverlay_FMF(imgToPrint,FMF_label, path.replace(args.dataType,'_segm_FMF.png')  )
       
        totalMuscle_label = mask == CLASS_MUSCLE
        fatTissueInMuscle_label = muscleQualityLabel == CLASS_MUSCLE_FAT
        muscleLowHu_label = muscleQualityLabel == CLASS_MUSCLE_LOW
        muscleHighHu_label = muscleQualityLabel == CLASS_MUSCLE
        muscleExclFat_label = muscleLowHu_label + muscleHighHu_label
        VAT_label = mask == CLASS_VISCERAL_FAT 
        SAT_label = mask == CLASS_SUBCUTAN_FAT 
        
        VAT_win_label, VAT_win_erod_label = createdWindowedAndErodedVisceralMasks(im, mask)
        labelToPrint_win = np.copy(mask); labelToPrint_win[mask==CLASS_VISCERAL_FAT] = 0; labelToPrint_win[VAT_win_label] = CLASS_VISCERAL_FAT
        labelToPrint_win_erod = np.copy(mask); labelToPrint_win_erod[mask==CLASS_VISCERAL_FAT] = 0; labelToPrint_win_erod[VAT_win_erod_label] = CLASS_VISCERAL_FAT
        printImageLabelOverlay(np.concatenate( (imgToPrint,imgToPrint,imgToPrint) ) ,
                               np.concatenate( (mask,labelToPrint_win,labelToPrint_win_erod) ) ,
                               path.replace(args.dataType,'_visc_erod.png'),
                               num_classes=len(SEGM_CLASSES))
         
        totalMuscleInclFat_numVoxel[idx] = np.sum( totalMuscle_label )
        totalMuscleExclFat_numVoxel[idx] = np.sum( muscleLowHu_label ) + np.sum( muscleHighHu_label )
        fatTissueInMuscle_numVoxel[idx] = np.sum( fatTissueInMuscle_label )
        muscleLowHu_numVoxel[idx] = np.sum( muscleLowHu_label )
        muscleHighHu_numVoxel[idx] = np.sum( muscleHighHu_label )
        SAT_numVoxel[idx] = np.sum( SAT_label )
        VAT_numVoxel[idx] = np.sum( VAT_label )
        VAT_win_numVoxel[idx] = np.sum( VAT_win_label )
        VAT_win_erod_numVoxel[idx] = np.sum( VAT_win_erod_label )
    
        if not (not factorNumToArea) and not math.isnan(factorNumToArea):
            totalMuscleInclFat_AreaInMM[idx] = factorNumToArea * totalMuscleInclFat_numVoxel[idx]
            totalMuscleExclFat_AreaInMM[idx] = factorNumToArea * totalMuscleExclFat_numVoxel[idx]
            fatTissueInMuscle_AreaInMM[idx] = factorNumToArea * fatTissueInMuscle_numVoxel[idx]
            muscleLowHu_AreaInMM[idx] = factorNumToArea * muscleLowHu_numVoxel[idx]
            muscleHighHu_AreaInMM[idx] = factorNumToArea * muscleHighHu_numVoxel[idx]
            SAT_AreaInMM[idx] = factorNumToArea * SAT_numVoxel[idx]
            VAT_AreaInMM[idx] = factorNumToArea * VAT_numVoxel[idx]
            VAT_win_AreaInMM[idx] = factorNumToArea * VAT_win_numVoxel[idx]
            VAT_win_erod_AreaInMM[idx] = factorNumToArea * VAT_win_erod_numVoxel[idx]
         
        totalMuscleInclFat_mean[idx] = np.mean(im[ totalMuscle_label ])
        totalMuscleInclFat_std[idx] = np.std( im[ totalMuscle_label ])
        totalMuscleExclFat_mean[idx] = np.mean(im[ muscleExclFat_label ])
        totalMuscleExclFat_std[idx] = np.std( im[ muscleExclFat_label ])
        fatTissueInMuscle_mean[idx] = np.mean(im[ fatTissueInMuscle_label ])
        fatTissueInMuscle_std[idx] = np.std( im[ fatTissueInMuscle_label ])
        muscleLowHu_mean[idx] = np.mean(im[ muscleLowHu_label ])
        muscleLowHu_std[idx] = np.std( im[ muscleLowHu_label ])
        muscleHighHu_mean[idx] = np.mean(im[ muscleHighHu_label ])
        muscleHighHu_std[idx] = np.std( im[ muscleHighHu_label ])
        SAT_mean[idx] = np.mean(im[ SAT_label ])
        SAT_std[idx] = np.std( im[ SAT_label ])
        VAT_mean[idx] = np.mean( im[ VAT_label ])
        VAT_std[idx] = np.std( im[ VAT_label ])
        
        VAT_win_mean[idx] = np.mean( im[ VAT_win_label ])
        VAT_win_std[idx] = np.std( im[ VAT_win_label ])
        VAT_win_erod_mean[idx] = np.mean( im[ VAT_win_erod_label ])
        VAT_win_erod_std[idx] = np.std( im[ VAT_win_erod_label ])
        
        muscleWithoutFat_numVoxel = (muscleHighHu_numVoxel[idx]+muscleLowHu_numVoxel[idx])
        if muscleWithoutFat_numVoxel!= 0:
            FatFreeMuscleFraction[idx] = muscleHighHu_numVoxel[idx] / muscleWithoutFat_numVoxel
            FattyMuscleFraction[idx] = muscleLowHu_numVoxel[idx] / muscleWithoutFat_numVoxel
            IntermuscleFatFraction[idx] = fatTissueInMuscle_numVoxel[idx] / muscleWithoutFat_numVoxel
        else:
            FatFreeMuscleFraction[idx] = 0
            FattyMuscleFraction[idx] = 0
            IntermuscleFatFraction[idx] = 0
              
    bodyComp_df = pd.DataFrame(dict(id=nii_names))
    bodyComp_df['QC_implants_det'] = QC_implants_det 
    bodyComp_df['QC_pred_dice_muscle_below_0_9'] = QC_pred_dice_muscle_below_0_9 
    bodyComp_df['QC_pred_dice_muscle'] = QC_pred_dice_muscle 
    bodyComp_df['QC_entropy_muscle'] = QC_entropy_muscle 
    bodyComp_df['FatFreeMuscleFraction'] = FatFreeMuscleFraction
    bodyComp_df['FattyMuscleFraction'] = FattyMuscleFraction
    bodyComp_df['IntermuscleFatFraction'] = IntermuscleFatFraction
    bodyComp_df['totalMuscleInclFat_AreaInMM'] = totalMuscleInclFat_AreaInMM
    bodyComp_df['totalMuscleExclFat_AreaInMM'] = totalMuscleExclFat_AreaInMM    
    bodyComp_df['muscleHighHu_AreaInMM'] = muscleHighHu_AreaInMM
    bodyComp_df['muscleLowHu_AreaInMM'] = muscleLowHu_AreaInMM
    bodyComp_df['fatTissueInMuscle_AreaInMM'] = fatTissueInMuscle_AreaInMM
    bodyComp_df['visceralFat_AreaInMM'] = VAT_AreaInMM
    bodyComp_df['visceralFat_windowed_AreaInMM'] = VAT_win_AreaInMM
    bodyComp_df['visceralFat_windowed_2eroded_AreaInMM'] = VAT_win_erod_AreaInMM
    bodyComp_df['subcutaneousFat_AreaInMM'] = SAT_AreaInMM
    bodyComp_df['pixelSpacing1'] = PixelSpacing1
    bodyComp_df['pixelSpacing2'] = PixelSpacing2
    bodyComp_df['totalMuscleInclFat_meanHU'] = totalMuscleInclFat_mean
    bodyComp_df['totalMuscleInclFat_stdHU'] = totalMuscleInclFat_std
    bodyComp_df['totalMuscleExclFat_meanHU'] = totalMuscleExclFat_mean
    bodyComp_df['totalMuscleExclFat_stdHU'] = totalMuscleExclFat_std
    bodyComp_df['muscleHighHu_meanHU'] = muscleHighHu_mean
    bodyComp_df['muscleHighHu_stdHU'] = muscleHighHu_std
    bodyComp_df['muscleLowHu_meanHU'] = muscleLowHu_mean
    bodyComp_df['muscleLowHu_stdHU'] = muscleLowHu_std
    bodyComp_df['fatTissueInMuscle_meanHU'] = fatTissueInMuscle_mean
    bodyComp_df['fatTissueInMuscle_stdHU'] = fatTissueInMuscle_std
    bodyComp_df['visceralFat_meanHU'] = VAT_mean
    bodyComp_df['visceralFat_stdHU'] = VAT_std
    bodyComp_df['visceralFat_windowed_meanHU'] = VAT_win_mean
    bodyComp_df['visceralFat_windowed_stdHU'] = VAT_win_std
    bodyComp_df['visceralFat_windowed_2eroded_meanHU'] = VAT_win_erod_mean
    bodyComp_df['visceralFat_windowed_2eroded_stdHU'] = VAT_win_erod_std
    bodyComp_df['subcutaneousFat_meanHU'] = SAT_mean
    bodyComp_df['subcutaneousFat_stdHU'] = SAT_std
    
    bodyComp_df.to_csv(os.path.join(args.inputPath+'_results.csv'), index=False, float_format='%.6f')
    bodyComp_df.to_excel(os.path.join(args.inputPath+'_results.xlsx'), index=False, float_format='%.6f')
                     
if __name__ == '__main__':
    main()











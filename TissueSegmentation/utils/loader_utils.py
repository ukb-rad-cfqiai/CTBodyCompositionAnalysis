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

import cv2
from .image_utils import imfill, getLargestContour
from fastai.vision import *
from fastai.vision.all import *
import nibabel as nib

def make_fp16(x):
    x=to_half(x)
    return x

# Cell
class TensorNifti(TensorImage):
    "Inherits from `TensorImage` and converts the `pixel_array` into a `TensorNifti`"
    _show_args = {'cmap':'gray'}

# Cell
class PILNifti(PILBase):
    _open_args,_tensor_cls,_show_args = {},TensorNifti,TensorNifti._show_args
    
    @classmethod
    def create(cls, fn:(Path,str,bytes), mode=None)->None:
        "Open a `Nifti file` from path `fn` or bytes `fn` and load it as a `PIL Image`"
        x = ((nib.load(str(fn))).get_fdata())
        
        im_shape = np.asarray(x.shape)
        new_shape = np.asarray([512,512])
        if any(im_shape[0:2] != new_shape):
            x = resize(x, new_shape, order=3, anti_aliasing=False, mode='constant', cval=0)

        x[x < (-1)*400] = (-1)*400
        x[x > 600] = 600
            
        x = np.array(x)
        bg_val = np.min(x)
        mask = np.uint8(x > bg_val)
        
        kernel_size_eyes = 10
        kernel_size = 5
        iters = 2
        kernel = np.uint8(np.ones((kernel_size,kernel_size)))

        mask = cv2.erode(mask,kernel,iterations = 1)
        mask = cv2.dilate(mask,kernel,iterations = 1)
        mask = imfill(mask)
        mask = cv2.erode(mask,kernel,iterations = iters)
        mask = getLargestContour(mask)
        mask = cv2.dilate(mask,kernel,iterations = iters)

        x[mask == 0] = bg_val
            
        x = (x-np.min(x))/(np.max(x)-np.min(x))
            
        
        if isinstance(fn,(Path,str)): im = Image.fromarray(x.squeeze())
        im.load()
        im = im._new(im.im)
        return cls(im.convert(mode) if mode else im)

PILNifti._tensor_cls = TensorNifti

def get_y_by_labelsFolder(x):
    return get_msk(str(x).replace('images','labels'))

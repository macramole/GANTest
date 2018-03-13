#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:18:44 2018

@author: leandro
"""

import numpy as np
from os import listdir, makedirs
import os.path
from keras.preprocessing.image import load_img, img_to_array
import h5py
import sys

def load_dataset(datasetDir, img_rows, img_cols):
    COUNT_NOTICE = 200
    imageSize = img_rows,img_cols
    
    print("loading", datasetDir, "...")
    df = None
    
    h5Path = os.path.join( datasetDir, "h5/")
    os.makedirs(h5Path, exist_ok=True)
    h5Path = os.path.join( h5Path, "dataset_" + str(img_rows) + ".h5")

    picDir = os.path.join( datasetDir, "pics/")

    if not os.path.isfile( h5Path ):
        print("No h5 dataset file found, processing images...")
        cantFiles = 0
        for f in listdir( picDir ):
            if f[-3:] == "jpg":
                cantFiles += 1
        
        if cantFiles == 0:
            print("No images found at %s" % picDir )
            exit()
        else:
            print("%d images found" % cantFiles )
        
        h5pyFile = h5py.File(h5Path, "w")
        df = h5pyFile.create_dataset("df", (cantFiles,img_rows,img_cols,3), dtype=int)
       
        i = 0
        for f in listdir( picDir ):
            if f[-3:] == "jpg":
                sys.stdout.write('.')
                sys.stdout.flush()
                
                img = load_img( os.path.join(picDir, f) )
                img = resize_and_crop(img, imageSize)
                img = img_to_array(img)
                df[i] = img
                
                i+=1
                
                if i % COUNT_NOTICE == 0:
                    sys.stdout.write('\n\r')
                    print("[", i, "/", cantFiles, "]")
                    sys.stdout.flush()
        
        print("H5 dataset file saved.")
    else:
        print("loading " + h5Path )
        h5File = h5py.File(h5Path, 'r')
        df = h5File["df"]

    return df

def resize_and_crop(img, size, crop_type='middle'):
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    #The image is scaled/cropped vertically or horizontally depending on the ratio
    if ratio > img_ratio:
        img = img.resize(( size[0], int(size[0] * img.size[1] / img.size[0]) )
                )
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (0, (img.size[1] - size[1]) / 2, img.size[0], (img.size[1] + size[1]) / 2)
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize(( int(size[1] * img.size[0] / img.size[1]), size[1]) )
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = ((img.size[0] - size[0]) / 2, 0, (img.size[0] + size[0]) / 2, img.size[1])
        elif crop_type == 'bottom':
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else :
        img = img.resize((size[0], size[1]) )
        # If the scale is the same, we do not need to crop
    
    return img
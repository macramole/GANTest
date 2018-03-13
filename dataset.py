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

def load_dataset(datasetDir, img_rows, img_cols):
    imageSize = img_rows,img_cols

    print("loading", datasetDir, "...")
    images = []
    
    npyPath = os.path.join( datasetDir, "npy/")
    os.makedirs(npyPath, exist_ok=True)
    npyPath = os.path.join( npyPath, "dataset_" + str(img_rows) + ".npy")

    picDir = os.path.join( datasetDir, "pics/")

    if not os.path.isfile( npyPath ):
        print("processing images...")
        cantFiles = 0
        for f in listdir( picDir ):
            if f[-3:] == "jpg":
                cantFiles += 1

                img = load_img( os.path.join(picDir, f) )
#                    img.thumbnail(imageSize)
                img = resize_and_crop(img, imageSize)
                img = img_to_array(img)

                images.append( img )
        
        if cantFiles == 0:
            print("No images found at %s" % picDir )
            exit()

        images = np.array(images)
        # print("shape images: ", images.shape)
        np.save( npyPath, images)
    else:
        print("loading " + npyPath )
        images = np.load( npyPath )

#    print("shape images: ", images.shape)

    return images

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
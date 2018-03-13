#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leandro
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset", default="-1", help="Path to dataset")
parser.add_argument("-s","--filesize", default="128x128x3", help="WIDTHxHEIGHT, must be a pow of 2, default: 128x128x3")
parser.add_argument("-t","--train", action="store_true")
parser.add_argument("-g","--generate", default="-1", help="Path to train directory")
parser.add_argument("--force-cpu", action="store_true")

parser.add_argument("-e", "--epochs", default=-1, type=int)
parser.add_argument("-bs", "--batch-size", default=32, type=int)
parser.add_argument("--save-interval-image", default=500, type=int)
parser.add_argument("--save-interval-model", default=1000, type=int)


args = parser.parse_args()

if not args.train and not args.generate:
    print("Please select train or generate")
    exit()

if args.train and not os.path.isdir(args.dataset):
    print("Dataset not found. Check your path")
    exit()

if args.force_cpu:
    print("Forcing CPU usage...")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
if args.train:
    arrInputSize = args.filesize.split("x")
    if len(arrInputSize) != 3:
        print("Invalid filesize!")
        exit()
    if int(arrInputSize[0]) % 2 != 0 or int(arrInputSize[1]) % 2 != 0:
        print("Input size must be power of 2")
        exit()

if args.train and args.epochs == -1:
    print("Define epoch count")
    exit()

import dcgan
import csv
from time import localtime, strftime

if args.train:
    trainPath = os.path.join(args.dataset, "train/", "dcgan_" + strftime("%Y-%m-%d_%H-%M-%S", localtime()) )
    os.makedirs( trainPath, exist_ok=True )
    
    with open( os.path.join(trainPath, "description.txt" ), "w" ) as descFile:
        descFile.write("Epochs: %d\n" % args.epochs )
        descFile.write("Batch size: %d\n" % args.batch_size )
        descFile.write("Input size: %s\n" % args.filesize )
    
    dcgan = dcgan.DCGAN( datasetDir = args.dataset, 
                        outputPath = trainPath,
                        img_rows = int(arrInputSize[0]),
                        img_cols = int(arrInputSize[1]),
                        channels = int(arrInputSize[2]),
                        )
    
    with open(os.path.join(trainPath,  "log.csv"), "w") as logfile:
        logwriter = csv.writer(logfile)
        logwriter.writerow(["epoch","d_loss", "d_acc", "g_loss"])

    dcgan.train(epochs=args.epochs, batch_size=args.batch_size, img_save_interval=args.save_interval_image, model_save_interval=args.save_interval_model)    
elif args.generate != "-1":
    if not os.path.isdir(args.generate):
        print("Path to train directory not found.")
        exit()
    
    datasetPath = os.path.join(args.generate, "../../" )
    modelPath = os.path.join(args.generate, "models/" )
    generatePath = os.path.join(datasetPath, "generate/" )
    os.makedirs(generatePath, exist_ok=True)
    
    epochs = inputSize = None
    with open( os.path.join(args.generate, "description.txt" ), "r" ) as descFile:
        line = descFile.readline()
        epochs = int( line[ line.find(":")+2: ] )
        descFile.readline()
        line = descFile.readline()
        inputSize = line[ line.find(":")+2: ]
    
    if args.epochs != -1:
        epochs = int( args.epochs )
    
    arrInputSize = inputSize.split("x")
    modelPath = os.path.join( modelPath, "generator.%d.h5" % epochs )
    
    dcgan = dcgan.DCGAN(datasetDir = datasetPath, 
                        outputPath = generatePath,
                        modelPath = modelPath,
                        img_rows = int(arrInputSize[0]),
                        img_cols = int(arrInputSize[1]),
                        channels = int(arrInputSize[2]),
                        )
    
    dcgan.generate()
    
    
    
    
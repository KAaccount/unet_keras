#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import json
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

from datagen import Dataset
from datagen import Dataloader
from model import Model


from lib.visualize import maskvisualizer
from lib.visualize import denormalize

from lib.augmentation import get_training_augmentation
from lib.augmentation import get_validation_augmentation
from lib.augmentation import get_preprocessing

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

plt.style.use('ggplot')
font = {'family' : 'meiryo'}
#matplotlib.rc('font', **font)
#plt.rcParams["figure.figsize"] = [10,10]



####
DATATYPE="voc"
ALLCLASSES= ["background","aeroplane", "bicycle", "bird","boat", "bottle", "bus",
            "car" , "cat", "chair", "cow",
            "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa","train", "tvmonitor"]
CLASSES  = ['horse','motorbike','person']
weights=[0.5,1,1,1]
jobid="20220729170654"

#####
"""
DATATYPE="cvd"
ALLCLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
            'tree', 'signsymbol', 'fence', 'car',
            'pedestrian', 'bicyclist', 'unlabelled']
CLASSES = ['car', 'pedestrian','building','pole','road']
weights=[1, 2, 1, 1, 1, 0.5]
jobid="20220729171039"
"""
#####


respath= "./result/"+jobid

train_imgfps=json.load(open(respath+'/train_imgfps.json', 'r'))
train_mskfps=json.load(open(respath+'/train_mskfps.json', 'r'))
valid_imgfps=json.load(open(respath+'/valid_imgfps.json', 'r'))
valid_mskfps=json.load(open(respath+'/valid_mskfps.json', 'r'))
test_imgfps=json.load(open(respath+'/test_imgfps.json', 'r'))
test_mskfps=json.load(open(respath+'/test_mskfps.json', 'r'))



dataset = Dataset(train_imgfps, train_mskfps, datatype=DATATYPE,allclasses=ALLCLASSES,
                classes=['person', "pottedplant"])
image, mask = dataset[10] # get some sample
maskvisualizer(resfilename=respath+"/checkA.png",image=image,
                bike_mask=mask[..., 0].squeeze(),sky_mask=mask[..., 1].squeeze(),
                background_mask=mask[..., 2].squeeze(),)

dataset = Dataset(train_imgfps, train_mskfps, datatype=DATATYPE,allclasses=ALLCLASSES,
                    classes=['horse', 'person'], augmentation=get_training_augmentation())
image, mask = dataset[12] # get some sample
maskvisualizer(resfilename=respath+"/checkB.png",image=image, cars_mask=mask[..., 0].squeeze(),
                sky_mask=mask[..., 1].squeeze(),background_mask=mask[..., 2].squeeze(),)





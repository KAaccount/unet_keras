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

import segmentation_models_org as sm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

plt.style.use('ggplot')
font = {'family' : 'meiryo'}
#matplotlib.rc('font', **font)
#plt.rcParams["figure.figsize"] = [10,10]

class Trainer(object):

    def __init__(self):

        ###voc
        self.DATATYPE="voc"
        self.ALLCLASSES= ["background","aeroplane", "bicycle", "bird","boat", "bottle", "bus", 
                    "car" , "cat", "chair", "cow", 
                    "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep", 
                    "sofa","train", "tvmonitor"]
        self.CLASSES  = ['horse','motorbike','person']
        self.weights=[0.5,1,1,1]
        self.jobid="20220729170654"
        """        
        ###cvd
        self.DATATYPE="cvd"
        self.ALLCLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
                    'tree', 'signsymbol', 'fence', 'car',
                    'pedestrian', 'bicyclist', 'unlabelled']
        self.CLASSES = ['car', 'pedestrian','building','pole','road']
        self.weights=[1, 2, 1, 1, 1, 0.5]
        self.jobid="20220729171039"
        """


        self.respath= "./result/"+self.jobid
        self.BACKBONE = 'efficientnetb3'
        self.LR = 0.0001
        self.BATCH_SIZE = 8
        self.EPOCHS = 3

        self.n_classes = 1 if len(self.CLASSES) == 1 else (len(self.CLASSES) + 1)  

            
        self.train_imgfps=json.load(open(self.respath+'/train_imgfps.json', 'r'))
        self.train_mskfps=json.load(open(self.respath+'/train_mskfps.json', 'r'))
        self.valid_imgfps=json.load(open(self.respath+'/valid_imgfps.json', 'r'))
        self.valid_mskfps=json.load(open(self.respath+'/valid_mskfps.json', 'r'))
        self.test_imgfps=json.load(open(self.respath+'/test_imgfps.json', 'r'))
        self.test_mskfps=json.load(open(self.respath+'/test_mskfps.json', 'r'))


    def __call__(self):

        preprocess_input = sm.get_preprocessing(self.BACKBONE)
        # Dataset for train images
        train_dataset = Dataset(
            self.train_imgfps, 
            self.train_mskfps, 
            datatype=self.DATATYPE,
            allclasses=self.ALLCLASSES, 
            classes=self.CLASSES, 
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocess_input),
        )

        # Dataset for validation images
        valid_dataset = Dataset(
            self.valid_imgfps, 
            self.valid_mskfps, 
            datatype=self.DATATYPE,
            allclasses=self.ALLCLASSES, 
            classes=self.CLASSES, 
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocess_input),
        )

        train_dataloader = Dataloader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        valid_dataloader = Dataloader(valid_dataset, batch_size=1, shuffle=False)

        # check shapes for errors
        assert train_dataloader[0][0].shape == (self.BATCH_SIZE, 320, 320, 3)
        assert train_dataloader[0][1].shape == (self.BATCH_SIZE, 320, 320, self.n_classes)

        print(train_dataloader[0][0].shape,train_dataloader[0][1].shape,len(train_dataloader))
        print(valid_dataloader[0][0].shape,valid_dataloader[0][1].shape,len(valid_dataloader))


        #create model
        ## 'Unet', 'PSPNet', 'FPN', 'Linknet',
        mdl=Model("Unet", self.BACKBONE ,self.n_classes,self.LR,self.weights)
        model=mdl()

        callbacks = [
            keras.callbacks.ModelCheckpoint(self.respath+'/best_model.h5', 
                                            save_weights_only=False, save_best_only=False, mode='min'),
            keras.callbacks.ReduceLROnPlateau(),
        ]

        # train model
        print("###train")
        history = model.fit_generator(
            train_dataloader, 
            steps_per_epoch=len(train_dataloader), 
            epochs=self.EPOCHS, 
            verbose=1,
            callbacks=callbacks, 
            validation_data=valid_dataloader, 
            validation_steps=len(valid_dataloader),
        )

        self.viewer(history)    


    def viewer(self,history):
        print(history.history.keys())

        # Plot training & validation iou_score values
        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        plt.plot(history.history['iou_score'])
        plt.plot(history.history['val_iou_score'])
        plt.title('Model iou_score')
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(122)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        #plt.show()
        plt.savefig(self.respath+f"/train.png")



if __name__=="__main__":
    trn=Trainer()
    trn()

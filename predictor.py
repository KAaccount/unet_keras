import json
import keras
import tensorflow as tf
import numpy as np

from datagen import Dataset
from datagen import Dataloader
from lib.augmentation import get_training_augmentation
from lib.augmentation import get_validation_augmentation
from lib.augmentation import get_preprocessing

from model import Model
from lib.visualize import maskvisualizer
from lib.visualize import denormalize

import segmentation_models_org as sm

class Predictor(object):


    def __init__(self):
        ####
        self.DATATYPE="voc"
        self.ALLCLASSES= ["background","aeroplane", "bicycle", "bird","boat", "bottle", "bus",
                    "car" , "cat", "chair", "cow",
                    "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                    "sofa","train", "tvmonitor"]
        self.CLASSES  = ['horse','motorbike','person']
        self.weights=[0.5,1,1,1]
        self.jobid="20220729170654"
        """

        #####
        self.DATATYPE="cvd"
        self.ALLCLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
                    'tree', 'signsymbol', 'fence', 'car',
                    'pedestrian', 'bicyclist', 'unlabelled']
        self.CLASSES = ['car', 'pedestrian','building','pole','road']
        self.weights=[1, 2, 1, 1, 1, 0.5]
        self.jobid="20220729171039"
        #####
        """


        self.BACKBONE = 'efficientnetb3'
        self.LR = 0.0001

        self.n_classes = 1 if len(self.CLASSES) == 1 else (len(self.CLASSES) + 1)  
        self.respath= "./result/"+self.jobid

        self.test_imgfps=json.load(open(self.respath+'/test_imgfps.json', 'r'))
        self.test_mskfps=json.load(open(self.respath+'/test_mskfps.json', 'r'))

    def __call__(self):
        preprocess_input = sm.get_preprocessing(self.BACKBONE)
        test_dataset = Dataset(
            self.test_imgfps,
            self.test_mskfps,
            datatype=self.DATATYPE,
            allclasses=self.ALLCLASSES,
            classes=self.CLASSES,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocess_input),
        )


        test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)


        mdl=Model("Unet", self.BACKBONE ,self.n_classes,self.LR,self.weights)
        model=mdl()


        # load best weights
        model.load_weights(self.respath+'/best_model.h5')
        scores = model.evaluate_generator(test_dataloader)

        print("Loss: {:.5}".format(scores[0]))
        #for metric, value in zip(metrics, scores[1:]):
        #    print("mean {}: {:.5}".format(metric.__name__, value))


        # # Visualization of results on test dataset
        n = 5
        ids = np.random.choice(np.arange(len(test_dataset)), size=n)

        for i in ids:

            image, gt_mask = test_dataset[i]
            image = np.expand_dims(image, axis=0)
            pr_mask = model.predict(image)

            maskvisualizer(
                resfilename=self.respath+f"/pred_{i}_a.png",
                image=denormalize(image.squeeze()),
                gt_mask=gt_mask.squeeze()[:,:,0:3],
                pr_mask=pr_mask.squeeze()[:,:,0:3],
            )
            maskvisualizer(
                resfilename=self.respath+f"/pred_{i}_b.png",
                image=denormalize(image.squeeze()),
                gt_mask=gt_mask.squeeze()[:,:,3:6],
                pr_mask=pr_mask.squeeze()[:,:,3:6],
            )

if __name__=="__main__":
    prd=Predictor()
    prd()


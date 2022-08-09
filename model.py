import keras
import tensorflow as tf
import numpy as np

import segmentation_models_org as sm


class Model(object):

    def __init__(self,mdltype,backbone,n_classes,LR,weights=None):

        self.mdltype=mdltype
        self.backbone=backbone
        self.n_classes = n_classes
        self.activation = 'sigmoid' if self.n_classes == 1 else 'softmax'
        self.LR=LR        
        self.weights=weights


        print("mdltype:  ",self.mdltype)
        print("backbone: ",self.backbone)
        print("n_classes:",self.n_classes)

        
    def __call__(self)  :
        ## 'Unet', 'PSPNet', 'FPN', 'Linknet',
        if self.mdltype == "Unet":
            model = sm.Unet(self.backbone, classes=self.n_classes, activation=self.activation) #OK
        elif self.mdltype == "PSPNet":
            model = sm.PSPNet(self.backbone, classes=self.n_classes, activation=self.activation) #NG
        elif self.mdltype == "FPN":
            model = sm.FPN(self.backbone, classes=self.n_classes, activation=self.activation) #OK
        elif self.mdltype == "Linknet":
            model = sm.Linknet(self.backbone, classes=self.n_classes, activation=self.activation) #OK
        else:
            print(f"mdltype {self.mdltype} is not allowed !!")
            exit()

        ### define optomizer
        optim = keras.optimizers.Adam(self.LR)

        ### Segmentation models losses 
        # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
        dice_loss = sm.losses.DiceLoss(class_weights=np.array(self.weights))
        focal_loss = sm.losses.BinaryFocalLoss() if self.n_classes == 1 else sm.losses.CategoricalFocalLoss()

        total_loss = dice_loss + (1 * focal_loss)
        # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
        #for metric, value in zip(metrics, scores[1:]):
        #    print("mean {}: {:.5}".format(metric.__name__, value))


        # compile keras model with defined optimozer, loss and metrics
        print("###compile")
        print("optim:",optim)
        print("total_loss:",total_loss)
        print("metrics:",metrics)
        model.compile(optim, total_loss, metrics)


        return model




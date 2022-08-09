import numpy as np
from PIL import Image
import cv2
"""
CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
           'tree', 'signsymbol', 'fence', 'car',
           'pedestrian', 'bicyclist', 'unlabelled']

class_values = [CLASSES.index(cls.lower()) for cls in classes

"""



tgt = '/work/DEVELOP/dataset/camvid/SegNet-Tutorial/CamVid/trainannot/0001TP_006690.png'

mask = cv2.imread(tgt, 0) #grayscale
print(mask.shape)

np.set_printoptions(threshold=np.inf)
image = Image.open(tgt)
image2 = np.asarray(image)
#print(image2)



tgt = '/work/DEVELOP/dataset/VOCdevkit/VOC2007/SegmentationClass/000063.png'

np.set_printoptions(threshold=np.inf)
image = Image.open(tgt)
image2 = np.asarray(image)
print(image2.shape)

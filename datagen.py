import numpy as np
import cv2 
from PIL import Image
import keras
import tensorflow as tf

class Dataset(object):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        datatype (str) : segmantation data type(voc, cvd)
        allclasses (list): values of allclasses 
        classes (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """


    def __init__(
            self,
            images_dir,
            masks_dir,
            datatype,
            allclasses,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.datatype=datatype
        self.images_fps=images_dir
        self.masks_fps=masks_dir

        # convert str names to class values on masks
        self.class_values = [allclasses.index(cls.lower()) for cls in classes]
        print("class value: ", self.class_values )

        self.augmentation = augmentation
        self.preprocessing = preprocessing
    def __getitem__(self, i):

        # read data
        # セグメンテーションのラベル番号と合わせてreadする必要あり
        # データセットごとに違うため要見直し
        if self.datatype=="voc":
            image = cv2.imread(self.images_fps[i])
            image = cv2.resize(image,(360,480))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = Image.open(self.masks_fps[i])
            mask = mask.resize((360,480))
            mask = np.asarray(mask)
        elif self.datatype=="cvd":
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[i], 0) #grayscale
        else:
            print(f"error : datatype {self.datatype} is not supported!")
            exit()


        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        #if self.preprocessing:
        #    sample = self.preprocessing(image=image, mask=mask)
        #    image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images_fps)


class Dataloader(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)



if __name__=="__main__":
    pass

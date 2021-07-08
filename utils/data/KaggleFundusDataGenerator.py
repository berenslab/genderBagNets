import keras
import numpy as np 
import pandas as pd
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from .preprocessing_utils import load_kaggle_fundus_img, circular_mask, subtract_bg_image
import os


class KaggleFundusDataGenerator(keras.utils.Sequence):
    def __init__(self, df, data_dir, batch_size=16, image_dimensions=(224,224,3), shuffle=False, augment=False, 
                 binary=True, balanced=False, kernel=None, kernel_size=None, mask=True, channel_first=False, preprocessing=None):
        self.data_dir = data_dir
        self.filenames = df['filename']
        self.labels = df['dr'] if binary else df['level']
        self.binary = binary
        self.balanced = balanced 
        self.indices = [] 

        if self.balanced:
            self.data = {}
            for i in np.arange(len(self.filenames)):
                if self.labels[i] not in self.data:
                    self.data[i] = []
                self.data[i].append(i) # adding index of file 

            self.classes = self.data.keys()
            batch_count = 0 
            counts_label = [0] * len(self.classes)
            
            for i in np.arange(self.__len__ * self.batch_size):
                for label in self.classes:
                    if current_index[label] == len(self.data[label]):
                        current_index[label] = 0
                    if counts_label[label] <= balanced_count:
                        self.indices.append(self.data[label][current_index[label]])
                        counts_label[label] += 1
                        current_index[label] += 1
                        batch_count += 1
                    if batch_count % 16 == 0:
                        batch_count = 0 
                        counts_label = [0] * len(self.classes)
                
        self.n_channels = image_dimensions[-1]
        self.dim = image_dimensions[:-1]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment 
        self.preprocessing = preprocessing
        self.kernel = kernel
        self.kernel_size = kernel_size
        self.mask = mask
        self.channel_first = channel_first
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.filenames) / self.batch_size))

    def on_epoch_end(self):
        if self.balanced and self.shuffle:
            raise ValueError('Both balanced and shuffle cannot be true at the same time')
        if self.balanced:
            self.indexes = self.indices
        else:
            self.indexes = np.arange(len(self.filenames))

            if self.shuffle:
                np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        #Balanced indices
        #indexes = 

        #Stratified sampler indices 
        #indexes = 
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # images, labels = self.__data_generation(indexes)
        labels = np.array([self.labels[k] for k in indexes])
        images = [cv2.imread(os.path.join(self.data_dir, self.filenames[k])) for k in indexes]
        images = [cv2.resize(images[i], (224,224)) for i in np.arange(len(images))]

        if self.augment:
            images = self.augmentor(images)

        images = np.array([self.fundus_preprocess(img) for img in images])
        images = np.squeeze(images, axis=1)

        return images, labels

    # def __data_generation(self, indexes):
    #     if self.channel_first:
    #         X = np.empty((self.batch_size, self.n_channels, *self.dim))
    #     else:
    #         X = np.empty((self.batch_size, *self.dim, self.n_channels))
    #     y = np.empty((self.batch_size), dtype=int)
        
    #     for i, ID in enumerate(indexes):
    #         image_filename = os.path.join(self.data_dir, self.filenames[ID])
    #         X[i, ] = load_kaggle_fundus_img(image_filename,
    #                                         img_size=self.dim,
    #                                         mask=self.mask, 
    #                                         kernel=self.kernel,
    #                                         ksize=self.kernel_size,
    #                                         preprocessing_function=self.preprocessing,
    #                                         channel_first=self.channel_first
    #                                        )
    #         y[i] = self.labels[ID]

    #     if not self.binary:
    #         targets = keras.utils.to_categorical(y, num_classes=self.n_classes)
    #     else:
    #         targets = y

    #     return X, targets

    def fundus_preprocess(self, image):
        h, w, c = image.shape
        #image = np.expand_dims(image, axis=0)
        if self.mask:
            image = circular_mask(image, h/2.0)

        if self.kernel is not None and self.kernel_size is not None:
            image = subtract_bg_image(image, kernel=self.kernel, ksize=self.kernel_size)

        if self.channel_first:
            image = np.moveaxis(image, -1, 1)

        if self.preprocessing is not None:
            image = self.preprocessing(image)

        return image
        

    def augmentor(self, images):
        aug_list = [iaa.Resize({"height":self.dim[0], "width":self.dim[1]}),
                    iaa.CropAndPad(percent=(0,0.15), pad_mode=["constant", "edge"],
                                   pad_cval=0.0),
                    iaa.MultiplyBrightness((0.5, 0.9)),
                    iaa.LinearContrast((0.5, 2.5)),
                    iaa.MultiplySaturation((0.5, 1.5)),
                    iaa.AddToHue((-40, 40)),
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.Affine(translate_percent={'x': (-0.048, 0.048), 'y': (-0.048, 0.048)},
                               rotate=(-15,15))]
        seq = iaa.Sometimes(p=0.9, then_list=aug_list)
        # seq = iaa.Sequential([iaa.Resize({"height":self.dim(0), "width":self.dim(1)}),
        #                       iaa.CropAndPad(percent=(0,0.15), pad_mode=["constant", "edge"],
        #                                      pad_cval=0.0),
        #                       iaa.MultiplyBrightness((0.5, 0.9)),
        #                       iaa.MultiplyContrast(0.5, 2.5),
        #                       iaa.MultiplySaturation(0.5, 1.5),
        #                       iaa.AddtoHue(-40, 40),
        #                       iaa.Fliplr(0.5),
        #                       iaa.Flipud(0.5),
        #                       iaa.Affine(transale_percent={'x': (-0.048, 0.048), 'y': (-0.048, 0.048)},
        #                                  rotate=(-15,15))])

        return seq.augment_images(images)

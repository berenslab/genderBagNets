import numpy as np
import keras
import os 

from .preprocessing_utils import load_ukb_fundus_img
from ..params import BATCH_SIZE, DEFAULT_IMG_SIZE, NUM_CLASSES, DEFAULT_NUM_CHANNELS
from ..paths import get_data_dir

class FundusDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=BATCH_SIZE,
                 dim=DEFAULT_IMG_SIZE, n_channels=DEFAULT_NUM_CHANNELS,
                 n_classes=NUM_CLASSES, shuffle=True,
                 data_aug=None, preprocessing_function=None, mask=True,
                 subtract_bg=True, kernel='median', kernel_size=23,
                 channel_first=False
                 ):
        self.dim = dim 
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels 
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = None 
        self.on_epoch_end()
        
        self.data_dir = get_data_dir(img_size=dim[0])
        self.data_aug = data_aug
        self.preprocessing_function = preprocessing_function 
        self.mask=mask
        self.subtract_bg = subtract_bg
        self.kernel = kernel 
        self.kernel_size = kernel_size
        self.channel_first = channel_first
        
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y 
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
            
    def __data_generation(self, list_IDs_temp):
        if self.channel_first:
            X = np.empty((self.batch_size, self.n_channels, *self.dim))
        else:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        for i, ID in enumerate(list_IDs_temp):
            image_filename = os.path.join(self.data_dir, ID)
            X[i, ] = load_ukb_fundus_img(image_filename,
                                         data_aug=self.data_aug,
                                         mask=self.mask, 
                                         subtract_bg=self.subtract_bg,
                                         kernel=self.kernel,
                                         ksize=self.kernel_size,
                                         preprocessing_function=self.preprocessing_function,
                                         channel_first=self.channel_first
                                        )
            y[i] = self.labels[ID]

        if self.n_classes > 1:
            targets = keras.utils.to_categorical(y, num_classes=self.n_classes)
        else:
            targets = y

        return X, targets

			

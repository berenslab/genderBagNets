import numpy as np
import cv2

from PIL import Image
from classification_models.keras import Classifiers

def get_preprocess_fn(DNN_type):
    if DNN_type == 'vgg16':
        from keras.applications.vgg16 import preprocess_input
        return preprocess_input
    elif DNN_type == 'vgg19':
        from keras.applications.vgg19 import preprocess_input
        return preprocess_input
    elif DNN_type in ['resnet18', 'resnet34', 'resnet50']:
        from keras.applications.resnet50 import preprocess_input
        return preprocess_input
    elif DNN_type == 'resnet101':
        from keras.applications.resnet101 import preprocess_input
        return preprocess_input
    elif DNN_type == 'resnet152':
        from keras.applications.resnet152 import preprocess_input
        return preprocess_input
    elif DNN_type == 'inceptionv3':
        from keras.applications.inception_v3 import preprocess_input
        return preprocess_input
    elif 'bagnet' in DNN_type:
        import numpy as np
        def preprocess_input(x):
            mean = np.expand_dims([0.485, 0.456, 0.406], axis=0)
            #For Kaggle
            # mean = np.expand_dims([0.45258693 0.29857974 0.20736021], axis=0)
            mean = np.expand_dims(np.expand_dims(mean, axis=-1), axis=-1)
            std = np.expand_dims([0.229, 0.224, 0.225], axis=0)
            std = np.expand_dims(np.expand_dims(std, axis=-1), axis=-1)

            #For Kaggle
            # std = np.expand_dims([0.27876022 0.1905769  0.14801926], axis=0)

            x = np.divide(x, 255.)

            x = np.subtract(x, mean)
            x = np.divide(x, std)

            return x
        return preprocess_input
    else:
        raise NotImplementedError

def normalize_img(img):	
    adjusted_std = np.maximum(np.std(img), 1.0/np.sqrt(np.prod(img.shape)))
    return np.divide(np.subtract(img, np.mean(img)), adjusted_std)

def circular_mask(img, radius):
    img = np.squeeze(img)

    radius = int(radius)
    b = np.zeros(img.shape)
    cv2.circle(b, (radius, radius), int(radius * 0.9),
               (1, 1, 1), -1, 8, 0)
    masked_img = img * b

    masked_img = np.expand_dims(masked_img, axis=0)
    return masked_img

def subtract_bg_image(img, kernel='median', ksize=23):
    img = np.squeeze(img)
    
    #if kernel == 'gauss':
    #    bg = cv2.GaussianBlur(img, (0, 0), ksize)
    #else:
    if ksize % 2 == 0:
        ksize = ksize + 1
    bg = cv2.medianBlur(img, ksize)
            
    img = cv2.addWeighted(img, 4, bg, -4, 128)
    img = np.expand_dims(img, axis=0)
    
    return img


def load_ukb_fundus_img(path, data_aug=None, 
                        mask=False, subtract_bg=False,
                        kernel='median', ksize=23, 
                        preprocessing_function=None,
                        channel_first=False
                       ):
    with Image.open(path) as img:	
        x = np.asarray(img)

    # print(x.shape)
    h, w, c = x.shape
        
    x = np.expand_dims(x, axis=0)
    
    if data_aug is not None:
        x = data_aug.flow(x, batch_size=1).next()
        x = x[0].astype('uint8')
        x = np.expand_dims(x, axis=0)
    #x = np.expand_dims(x, axis=0)
        
    if mask:
        # print('mask')
        x = circular_mask(x, h/2.0)

    if subtract_bg:
        # print('sub bg')
        x = subtract_bg_image(x, kernel=kernel, ksize=ksize)
        
    if channel_first:
        # print('ch first')
        x = np.moveaxis(x, -1, 1)
        
    if preprocessing_function is not None:
        x = preprocessing_function(x)
        
    return x

def load_kaggle_fundus_img(path, img_size=(224,224), mask=False, 
                           kernel=None, ksize=None,                         
                           channel_first=False, 
                           preprocessing_function=None):

    x = cv2.imread(path)
    # print(x.shape)
    x = cv2.resize(x, img_size)
    h, w, c = x.shape
        
    # x = np.expand_dims(x, axis=0)
        
    if mask:
        # print('mask')
        x = circular_mask(x, h/2.0)

    if kernel is not None and ksize is not None:
        # print('sub bg')
        x = subtract_bg_image(x, kernel=kernel, ksize=ksize)
        
    if channel_first:
        # print('ch first')
        x = np.moveaxis(x, -1, 1)
        
    if preprocessing_function is not None:
        x = preprocessing_function(x)
        
    return x


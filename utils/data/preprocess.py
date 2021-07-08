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
            mean = np.expand_dims(np.expand_dims(mean, axis=-1), axis=-1)
            std = np.expand_dims([0.229, 0.224, 0.225], axis=0)
            std = np.expand_dims(np.expand_dims(std, axis=-1), axis=-1)

            x = np.divide(x, 255.)

            x = np.subtract(x, mean)
            x = np.divide(x, std)

            return x
        return preprocess_input
    else:
        raise NotImplementedError

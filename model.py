from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Activation, Dropout, Concatenate, Flatten
from keras import regularizers
from utils.params import DEFAULT_DNN_TYPE, DEFAULT_INIT, REG_CONSTANT, DEFAULT_NUM_CHANNELS, NUM_CLASSES, DEFAULT_IMG_SIZE
from classification_models.keras import Classifiers

def get_base_network(DNN_type):
    if DNN_type == 'vgg16':
        from keras.applications.vgg16 import VGG16
        from keras.applications.vgg16 import preprocess_input
        Network = VGG16
    elif DNN_type == 'vgg19':
        from keras.applications.vgg19 import VGG19
        from keras.applications.vgg19 import preprocess_input
        Network = VGG19
    elif DNN_type == 'resnet18':
    	Network, _ = Classifiers.get('resnet18')
    elif DNN_type == 'resnet34':
    	Network, _ = Classifiers.get('resnet34')
    elif DNN_type == 'resnet50':
        from keras.applications.resnet50 import ResNet50
        from keras.applications.resnet50 import preprocess_input
        Network = ResNet50
    elif DNN_type == 'resnet101':
        from keras.applications.resnet101 import ResNet101
        from keras.applications.resnet101 import preprocess_input
        Network = ResNet101
    elif DNN_type == 'resnet152':
        from keras.applications.resnet152 import ResNet152
        from keras.applications.resnet152 import preprocess_input
        Network = ResNet152
    elif DNN_type == 'inceptionv3':
        from keras.applications.inception_v3 import InceptionV3
        from keras.applications.inception_v3 import preprocess_input
        Network = InceptionV3
    elif DNN_type == 'bagnet9':
    	from bagnets.kerasnet import bagnet9
    	Network = bagnet9
    elif DNN_type == 'bagnet17':
    	from bagnets.kerasnet import bagnet17
    	Network = bagnet17
    elif DNN_type == 'bagnet33':
    	from bagnets.kerasnet import bagnet33
    	Network = bagnet33
    else:
        raise NotImplementedError
    return Network 

def add_regularizers(base_model, DNN_type=DEFAULT_DNN_TYPE):
    for i in range(len(base_model.layers)):
        layer = base_model.layers[i]
        print("%d\t%s\t%s" % (i, layer.name, layer.__class__.__name__))
        
        if (layer.__class__.__name__ == 'Conv2D' or layer.__class__.__name__ == 'Dense'):
            if layer.kernel_regularizer == None:
                print(type(layer.kernel_regularizer))
                print('No regularizer found! Adding a regulaizer')
                layer.kernel_regularizer = regularizers.l2(REG_CONSTANT)
                print(layer.kernel_regularizer.get_config())
                
        if i < len(base_model.layers)-1 and (DNN_type == 'vgg16' or DNN_type == 'vgg19'):
            base_model.layers[i+1].inputs = BatchNormalization()(layer.output)
            
    print('DONE with checking the regularizers in the base model')
	

def get_base_model(DNN_type=DEFAULT_DNN_TYPE, base_model_init=DEFAULT_INIT, num_channels=DEFAULT_NUM_CHANNELS, num_classes=NUM_CLASSES,
                   input_dims=(DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE)):
    Network = get_base_network(DNN_type=DNN_type)
    if 'bagnet' in DNN_type:
    	print('Bagnet')
    	base_model = Network()
    else:
    	base_model = Network(include_top=False, weights=base_model_init,
                         	 input_tensor=None, input_shape=(input_dims + (num_channels,)),
                         	 pooling=None, classes=num_classes
                        	)
    
    if DNN_type == 'vgg16':
        base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)
    elif DNN_type == 'vgg19':
        base_model = Model(input=base_model.input, outputs=base_model.get_layer('block5_conv4').output)
    elif DNN_type == 'bagnet9':
    	base_model = Model(input=base_model.input, outputs=base_model.get_layer('layer4.2.relu0.4003240620620412').output)
    elif DNN_type == 'bagnet17':
    	base_model = Model(input=base_model.input, outputs=base_model.get_layer('layer4.2.relu0.6627657152906141').output)
    elif DNN_type == 'bagnet33':
    	base_model = Model(input=base_model.input, outputs=base_model.get_layer('layer4.2.relu0.23004603898000509').output)
    else:
        base_model = None
        
        
    add_regularizers(base_model, DNN_type)
    
    base_model = Model(inputs=base_model.input, outputs=base_model.output)
    
    return base_model


def get_top_model_outputs(x, num_classes=NUM_CLASSES):
	x_avg = GlobalAveragePooling2D()(x)
	x_max = GlobalMaxPooling2D()(x)
	x = Concatenate()([x_avg, x_max])

	x = Dense(2048, kernel_regularizer=regularizers.l2(REG_CONSTANT))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Dense(512, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(REG_CONSTANT))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Dense(num_classes, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(REG_CONSTANT))(x)
	if num_classes == 1:
		outputs = Activation('sigmoid')(x)
	else:
		outputs = Activation('softmax')(x)

	return outputs

def get_top_model_for_bagnet(x, num_classes=NUM_CLASSES):
	#x = base_model.get_layer('layer4.2.relu0.23004603898000509').output
	x_avg = GlobalAveragePooling2D(data_format='channels_first')(x)

	x = Dense(num_classes, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(REG_CONSTANT))(x_avg)
	if num_classes == 1:
		outputs = Activation('sigmoid')(x)
	else:
		outputs = Activation('softmax')(x)

	return outputs

def build_model(DNN_type=DEFAULT_DNN_TYPE, base_model_init=DEFAULT_INIT, num_channels=DEFAULT_NUM_CHANNELS, num_classes=NUM_CLASSES,
                input_dims=(DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE)):
    base_model = get_base_model(DNN_type, base_model_init, num_channels, num_classes, input_dims)
    if 'bagnet' in DNN_type:
    	print('Bagnet model')
    	top_model_outputs = get_top_model_for_bagnet(base_model.output, num_classes)
    else:
    	top_model_outputs = get_top_model_outputs(base_model.output, num_classes)
    
    model = Model(inputs=base_model.input, output=top_model_outputs)
    print(model.summary())
    
    return base_model, model






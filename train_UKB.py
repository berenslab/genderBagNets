from keras import optimizers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy

from utils.data.FundusDataGenerator import FundusDataGenerator 
from utils.data.load_UKB import get_dataset
from utils.data.preprocess import get_preprocess_fn
from utils import paths
from utils import params
from utils.plot_utils.plot_fns import plot_history

import model
import argparse

def train(base_DNN_type=params.DEFAULT_DNN_TYPE, img_size=params.DEFAULT_IMG_SIZE, batch_size=params.BATCH_SIZE, 
		  learning_rate=params.DEFAULT_LEARNING_RATE,fc_epochs=params.FC_EPOCHS, epochs=params.EPOCHS, mask=params.MASK,
		  subtract_bg=params.SUBTRACT_BG, kernel=params.DEFAULT_KERNEL, kernel_size=params.DEFAULT_KERNEL_SIZE, 
		  train_ratio=params.TRAIN_RATIO, val_ratio=params.VAL_RATIO, num_classes=params.NUM_CLASSES, 
		  num_channels=params.DEFAULT_NUM_CHANNELS, num_workers=params.NUM_WORKERS, 
		  max_queue_size=params.MAX_QUEUE_SIZE, label_smoothing=0.0, channel_first=params.CHANNEL_FIRST):
    
	input_dims = (img_size, img_size)	
	data, labels = get_dataset(train_ratio=train_ratio, val_ratio=val_ratio)
	#undersampled_data, labels = get_dataset(train_ratio=train_ratio, val_ratio=val_ratio, undersampling_rate=0.05, oversampling_rate=None)

	datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, 
	                             featurewise_std_normalization=False, samplewise_std_normalization=False,
	                             zca_whitening=False, zca_epsilon=1e-06,
	                             rotation_range=45,
	                             width_shift_range=30, height_shift_range=30, 
	                             brightness_range=(0.9, 1.1),
	                             shear_range=0.0, zoom_range=0.1,
	                             channel_shift_range=0.0, fill_mode='nearest',
	                             cval=0.0, horizontal_flip=True, vertical_flip=True, 
	                             rescale=None, validation_split=0.0, dtype=None
	                            )

	preprocessing_function = get_preprocess_fn(base_DNN_type)
        
	#init_train_datagen = FundusDataGenerator(undersampled_data['train'], labels,
	#                                    batch_size=batch_size, dim=input_dims, 
	#                                    n_channels=num_channels, n_classes=params.NUM_CLASSES, shuffle=True,
	#                                    data_aug=datagen, preprocessing_function=preprocessing_function,
	#                                    subtract_bg=subtract_bg, kernel=kernel, kernel_size=kernel_size 
	#                                   )


	train_datagen = FundusDataGenerator(data['train'], labels,
	                                    batch_size=batch_size, dim=input_dims, 
	                                    n_channels=num_channels, n_classes=num_classes, shuffle=True,
	                                    data_aug=datagen, preprocessing_function=preprocessing_function, mask=mask,
	                                    subtract_bg=subtract_bg, kernel=kernel, kernel_size=kernel_size,
	                                    channel_first=channel_first
	                                   )


	val_datagen = FundusDataGenerator(data['val'], labels,
	                                  batch_size=batch_size, dim=input_dims, 
	                                  n_channels=num_channels, n_classes=num_classes, shuffle=True,
	                                  data_aug=datagen, preprocessing_function=preprocessing_function, mask=mask, 
	                                  subtract_bg=subtract_bg, kernel=kernel, kernel_size=kernel_size,
	                                  channel_first=channel_first
	                                 )

	adam = optimizers.SGD(lr=learning_rate)
	# if label_smoothing > 0.0:
	# 	loss_fn = CategoricalCrossentropy(label_smoothing=label_smoothing)
	# else:
	# 	loss_fn = CategoricalCrossentropy()

	base_model, fundus_model = model.build_model(DNN_type=base_DNN_type, num_channels=num_channels,
	                                             num_classes=num_classes, input_dims=input_dims    
	                                            )


	ckpt_path = paths.get_checkpoint_path(dnn_type=base_DNN_type)

	for layer in base_model.layers:
		layer.trainable = False


	fundus_model.compile(optimizer=adam, loss=BinaryCrossentropy(label_smoothing=label_smoothing), metrics=['acc'])
	checkpoint = ModelCheckpoint(ckpt_path, monitor='val_acc', verbose=1, save_best_only=False, mode='max', period=1) 

	callbacks_list = [checkpoint]

	history_1 = fundus_model.fit_generator(generator=train_datagen, validation_data=val_datagen,
	                                       epochs=fc_epochs, use_multiprocessing=True, workers=num_workers, verbose=1,	
	                                       callbacks=callbacks_list, max_queue_size=max_queue_size, initial_epoch=0
	                                       #class_weight={0:0.04, 1:0.96}
	                                      )

	for layer in base_model.layers:
		layer.trainable = True

	fundus_model.compile(optimizer=adam, loss=BinaryCrossentropy(label_smoothing=label_smoothing), metrics=['acc'])
	
        #Train first with undersampling
	history_2 = fundus_model.fit_generator(generator=train_datagen, validation_data=val_datagen,
	                                       epochs=epochs, use_multiprocessing=True, workers=num_workers, verbose=1,	
	                                       callbacks=callbacks_list, max_queue_size=max_queue_size, initial_epoch=0 
                                               #class_weight={0:0.04, 1:0.96}
	                                      )
        #Now with oversampling
	#history_3 = fundus_model.fit_generator(generator=train_datagen, validation_data=val_datagen,
	#                                       epochs=epochs-80, use_multiprocessing=True, workers=num_workers, verbose=1,	
	#                                       callbacks=callbacks_list, max_queue_size=max_queue_size, initial_epoch=0 
        #                                       #class_weight={0:0.04, 1:0.96}
	#                                      )

        
	del(fundus_model)
	save_hist_path = paths.get_history_path(dnn_type=base_DNN_type)
	plot_history([history_1, history_2], savepath=save_hist_path)
	#evaluate_model(data, labels) -> to generate evaluation rpert from saved model


if __name__=='__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--BASE_MODEL_TYPE', default=params.DEFAULT_DNN_TYPE, help='Base network type. Default: %s' % (params.DEFAULT_DNN_TYPE))
	parser.add_argument('--IMG_SIZE', default=params.DEFAULT_IMG_SIZE, help='Image dimensions. Possible values: 224, 256, 512, 587. \
						Default value: %d' % (params.DEFAULT_IMG_SIZE))
	parser.add_argument('--BATCH_SIZE', default=params.BATCH_SIZE, help='Mini batch size. Default value: %d' % (params.BATCH_SIZE))
	parser.add_argument('--FC_EPOCHS', default=params.FC_EPOCHS, help='Number of epochs for warm start. Default_value: %d' % (params.FC_EPOCHS))
	parser.add_argument('--EPOCHS', default=params.EPOCHS, help='Number of epochs for training full model after warm start.\
						 Default value: %d' % (params.EPOCHS))
	parser.add_argument('--LEARNING_RATE', default=params.DEFAULT_LEARNING_RATE, help='Learning rate. Default value: %f' % (params.DEFAULT_LEARNING_RATE))
	parser.add_argument('--NUM_CLASSES', default=params.NUM_CLASSES, help='Number of distict classes to classify into. Default value: %s' % (params.NUM_CLASSES))
	parser.add_argument('--NUM_CHANNELS', default=params.DEFAULT_NUM_CHANNELS, help='Number of input channels. Default value: %s' % (params.DEFAULT_NUM_CHANNELS))
	parser.add_argument('--NUM_WORKERS', default=params.NUM_WORKERS, help='Number of parallel workers to be used. Default value: %s' % (params.NUM_WORKERS))
	parser.add_argument('--MAX_QUEUE_SIZE', default=params.MAX_QUEUE_SIZE, help='Maximum queue size. Default value: %s' % (params.MAX_QUEUE_SIZE))
	parser.add_argument('--LABEL_SMOOTHING', default=0., help='Weights for label smoothing in the crossentropy loss function')
	parser.add_argument('--CHANNEL_FIRST', type=int, default=False, help='True if the first dimension of input corresponds to the channel')
	parser.add_argument('--SUBTRACT_BG', type=int, default=False, help='True if background subtraction needs to be enabled')
	parser.add_argument('--MASK', type=int, default=False, help='True if circular masking needs to be enabled')

	args = parser.parse_args()

	train(base_DNN_type=args.BASE_MODEL_TYPE, img_size=int(args.IMG_SIZE), batch_size=int(args.BATCH_SIZE), 
	      learning_rate=float(args.LEARNING_RATE), fc_epochs=int(args.FC_EPOCHS), epochs=int(args.EPOCHS), mask=bool(args.MASK),
	      subtract_bg=bool(args.SUBTRACT_BG), num_classes=int(args.NUM_CLASSES), num_channels=int(args.NUM_CHANNELS), 
	      num_workers=int(args.NUM_WORKERS), max_queue_size=int(args.MAX_QUEUE_SIZE),
	      label_smoothing=float(args.LABEL_SMOOTHING), channel_first=bool(args.CHANNEL_FIRST))


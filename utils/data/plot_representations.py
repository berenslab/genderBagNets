from keras.models import load_model
import numpy as np 
import os 
import argparse

from utils.paths import get_model_path, get_data_dir, get_model_path, get_roc_path
from utils.params import TRAIN_RATIO, VAL_RATIO, DEFAULT_IMG_SIZE, DEFAULT_KERNEL, DEFAULT_KERNEL_SIZE, DEFAULT_DNN_TYPE 
from utils.data.load_UKB import get_dataset 
from utils.data.UKB_utils import load_ukb_fundus_img
from utils.plot_utils.plot_fns import plot_roc_auc, plot_confusion_matrix_all
from utils.data.preprocess import get_preprocess_fn

def extract_features(model, partition, labels):
	layer_name = ''
	batch_size = 4

	model = load_model(model_path)
	rep_layer_outs = model.get_layer(layer_name).output
	rep_model = Model(inputs=model.input, outputs=rep_layer_outputs)

	datagen = FundusDataGenerator(partition, labels,
								  batch_size=batch_size, dim=input_dims, 
								  n_channels=num_channels, n_classes=params.NUM_CLASSES, shuffle=True,
								  data_aug=datagen, preprocessing_function=preprocessing_function,
								  subtract_bg=subtract_bg, kernel=kernel, kernel_size=kernel_size 
	                             )
	features = rep_model.predict_generator(datagen, steps=int(len(partition))//batch_size)
	features = features.reshape(*features.shape[:1], -2)

	return features, labels

def plot_features():
	data, labels = get_dataset(train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)
	features_tr, labels_tr = extract_features(model, data['train'], labels)
	
	#plot each set separately 


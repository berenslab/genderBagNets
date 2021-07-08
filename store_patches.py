from tables import *
import numpy as np 
import pandas as pd

import tensorflow as tf
import keras
from keras import backend as K
import argparse
#print(K.image_data_format())
K.set_image_data_format('channels_first')

from keras.models import Model
from keras.layers import Input, Dense
from keras.models import load_model

import collections
import os
from PIL import Image

from utils.data.load_UKB import get_dataset
from utils.params import TRAIN_RATIO, VAL_RATIO
from utils.data.UKB_utils import load_ukb_fundus_img
from utils.data.preprocess import get_preprocess_fn
from utils.paths import get_model_path, get_data_dir

def sigmoid(x):
        return  np.where(x >= 0, 1 / (1 + np.exp(-x)), 
                         np.exp(x) / (1 + np.exp(x)))

class Patch(IsDescription):
	filename = StringCol(1024)
	global_index = Int64Col()
	local_index = Int64Col()
	true_label = BoolCol()
	predicted_label = BoolCol()
	# female_logit = Float32Col()
	# male_logit = Float32Col()
	logit = Float32Col()
	local_evidence_female = Int64Col()
	local_evidence_male = Int64Col()
	prob = Float32Col()
	# female_prob = Float32Col()
	# male_prob = Float32Col()
	left_right = BoolCol()
	row_id = Int32Col()
	col_id = Int32Col()
	global_quality = Float32Col()

def writerow(patch, attr):
	patch['filename'] = attr['filename']
	patch['global_index'] = attr['global_index']
	patch['local_index'] = attr['local_index']
	patch['true_label'] = attr['true_label']
	patch['predicted_label'] = attr['predicted_label']
	# patch ['female_logit'] = attr['female_logit']
	# patch['male_logit'] = attr['male_logit']
	patch['logit'] = attr['logit']
	patch['local_evidence_female'] = attr['local_evidence_female']
	patch['local_evidence_male'] = attr['local_evidence_male']
	# patch['female_prob'] = attr['female_prob']
	# patch['male_prob'] = attr['male_prob']
	patch['prob'] = attr['prob']
	patch['left_right'] = attr['left_right']
	patch['row_id'] = attr['i']
	patch['col_id'] = attr['j']
	patch['global_quality'] = attr['global_quality']
	patch.append()

def get_models(model_filename, dnn_type='bagnet33'):
	model_dir = '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/supervised/modelstore/%s/' % (dnn_type)
	model_filepath = os.path.join(model_dir, model_filename)

	bagnet = load_model(model_filepath)
	print('========================================================================')
	print('BagNet')
	bagnet.summary()
	print('========================================================================')

	# feature extractor 
	feature_extractor = Model(inputs=bagnet.input, outputs=bagnet.get_layer('layer4.2.relu0.23004603898000509').output)
	for layer in feature_extractor.layers:
		layer.trainable = False
	print('========================================================================')
	print('Feature Extractor')
	feature_extractor.summary()
	print('========================================================================')

	feature_vector = Input(shape=(2048,))
	logits = bagnet.get_layer('dense_1')(feature_vector)
	softmax = bagnet.get_layer('activation_1')(logits)

	logit_machine = Model(inputs=feature_vector, outputs=logits)
	for layer in logit_machine.layers:
		layer.trainable = False
	print('========================================================================')
	print('Logit Machine')
	logit_machine.summary()
	print('========================================================================')

	lin_classifier = Model(inputs=feature_vector, outputs=softmax)
	for layer in lin_classifier.layers:
		layer.trainable = False
	print('========================================================================')    
	print('Linear Classifier')
	lin_classifier.summary()
	print('========================================================================')

	return bagnet, feature_extractor, logit_machine, lin_classifier

def create_storage_hierarchy(data, labels, partition, data_dir, 
							 qual_df, models, preprocess_fn, mask=True, 
							 subtract_bg=True, kernel='median',
							 kernel_size=23, channel_first=True):
	num_images = len(data[partition])

	#Pytables data hierachy specification
	storage_dir = '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/supervised/patch_features/'
	filepath = os.path.join(storage_dir, "bagnet33_patch_attributes_and_features_%s_v4.h5"%(partition))
	h5file = open_file(filepath, mode="w", title="Patch descriptor")
	patch_group = h5file.create_group("/", 'patch', 'Patch information')
	patch_table = h5file.create_table(patch_group, 'patchinfo', Patch, 'Patch attributes')
	patch = patch_table.row

	num_patches = num_images * 24 * 24
	feature_dims = 2048

	shape = (num_patches, feature_dims)
	atom = Float32Atom()
	filters = Filters(complevel=5, complib='zlib')

	features_group = h5file.create_group("/", 'patch_features', 'Patch features array')
	patch_features = h5file.create_carray(features_group, 'features', atom, shape, 
										  filters=filters)

	#Specifying the models
	bagnet, feature_extractor, logit_machine, lin_classifier = models


	#Extracting patch features and attributes and storing in h5 files.
	patch_idx = 0 
	sentinel = 1
	
	for filename in data[partition]:
		local_patch_idx = 0 
		gender = int(labels[filename])
		global_qual = qual_df.loc[filename]['ungradable_prob_median']
		if '21015' in filename:
			left_right = 'l'
		else:
			left_right = 'r'

		path = os.path.join(data_dir, filename)
		img = load_ukb_fundus_img(path=path, data_aug=None, mask=mask, 
		                          subtract_bg=subtract_bg, kernel=kernel, 
		                          ksize=kernel_size, 
		                          channel_first=channel_first,
		                          preprocessing_function=preprocess_fn)    

		# use BagNet to obtain a global decision for the image
                #If using softmax instead of sigmoid 
		# softmax_out = bagnet.predict(img)[0]
		# gender_pred = np.argmax(softmax_out, axis=-1)

		sigmoid_out = bagnet.predict(img)[0]
		gender_pred = (sigmoid_out > 0.5)

		# extract the features with a BagNet instance
		features = feature_extractor.predict(img)
	    
	    # generate a coarse (24x24) heatmap
		for i in range(features.shape[-2]): # heatmap.shape[0]
			for j in range(features.shape[-1]): # heatmap.shape[1]
				logits_ij = logit_machine.predict(features[:,:,i,j])[0]
				logit = logits_ij[0]

                                # If using softmax instead of sigmoid 
				# female_logit = logits_ij[0]
				# male_logit = logits_ij[1]
				# softmax_op = lin_classifier.predict(features[:,:,i,j])[0]
				# female_prob = softmax_op[0]
				# male_prob = softmax_op[1]
				# local_evidence_female = int(female_prob > 0.5)
				# local_evidence_male = int(male_prob > 0.5)

				prob = lin_classifier.predict(features[:,:,i,j])[0]


				local_evidence_female = int(prob < 0.5)
				local_evidence_male = int(prob >= 0.5)

				patch_features[patch_idx, :] = features[:,:,i,j]

                                # If using softmax instead of sigmoid 
				# writerow(patch, {'filename':str(filename), 'global_index': patch_idx, 
				# 				 'local_index':local_patch_idx, 'true_label':bool(gender), 
				# 				 'predicted_label':bool(gender_pred),'female_logit':logits_ij[0], 
				# 				 'male_logit':logits_ij[1], 'local_evidence_female':local_evidence_female, 
				# 				 'local_evidence_male':local_evidence_male, 'female_prob': female_prob,
				# 				  'male_prob':male_prob, 'left_right':left_right, 'i':i, 
				# 				 'j':j,'global_quality':global_qual}) 


				writerow(patch, {'filename':str(filename), 'global_index': patch_idx, 
								 'local_index':local_patch_idx, 'true_label':bool(gender), 
								 'predicted_label':bool(gender_pred), 
								 'logit':logit, 'local_evidence_female':local_evidence_female, 
								 'local_evidence_male':local_evidence_male, 'prob': prob,
								 'left_right':left_right, 'i':i, 
								 'j':j,'global_quality':global_qual}) 
				patch_idx += 1
				local_patch_idx += 1

		if sentinel%2000 == 0:
			patch_table.flush()
			print('%d out of %d test images processed.' % (sentinel, len(data[partition])))
		sentinel = sentinel + 1

	h5file.close()
	return -1


def test_store(partition):
	storage_dir = '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/supervised/patch_features/'
	filepath = os.path.join(storage_dir, "bagnet33_patch_attributes_and_features_%s_v4.h5"%(partition))
	h5file = open_file(filepath, mode="r")
	print('Number of images derived from table: %d'%(h5file.root.patch.patchinfo.nrows/576))
	print('Number of images derived from feature array: %d'%(h5file.root.patch_features.features.shape[0]/576))
	print('========================================================================')
	return -1

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_filename', type=str, help='Checkpoint path of the model')
	parser.add_argument('--subtract_bg', type=int, help='If 1 background is subtracted from image, if 0 backgroud is not subtracted')
	parser.add_argument('--mask', type=int, help='If 1 circular mask is applied to input image, if 0 circular mask function not applied')

	args = parser.parse_args()

	dnn_type = 'bagnet33'
	data_dir = get_data_dir(img_size=224)
	preprocess_fn = get_preprocess_fn(dnn_type)
	
	data, labels = get_dataset(train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)

	qual_file = "/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/qualfiles/ukb_fundus_EyeQual_Ensemble_ISBI2020_v3.csv"
	qual_df = pd.read_csv(qual_file) 
	qual_df = qual_df[['filename', 'ungradable_prob_median']]
	qual_df.set_index("filename", inplace=True)

	models = get_models(args.model_filename, 'bagnet33')

	for partition in ['val', 'test', 'train']:
		print('========================================================================')
		print('Processing %s partition'%(partition))
		create_storage_hierarchy(data, labels, partition, data_dir, qual_df, models, preprocess_fn, mask=bool(args.mask), subtract_bg=bool(args.subtract_bg))
		test_store(partition)

from tables import *
import numpy as np 
import os

from utils.tsne_utils import plot_tsne, save_tsne_points

def filter_patches(h5file, qual_threshold, condition, logit_key):
    patch_table = h5file.root.patch.patchinfo
    indices_and_labels = [(patch['global_index'], patch[logit_key]) for patch in patch_table.where(condition)]
    return indices_and_labels


def get_true_labels_and_features(storage_dir, h5filename, quality_threshold, condition, logit_key):
    h5filepath = os.path.join(storage_dir, h5filename)
    h5file = open_file(h5filepath, "r")
    
    indices_and_labels = filter_patches(h5file, quality_threshold, condition, logit_key)
    print('Number of patches filtered: %d'%(len(indices_and_labels)))

    indices = [i for (i, _) in indices_and_labels]
    features = h5file.root.patch_features.features[indices, :]
    labels = [l for (_,l) in indices_and_labels]
    return features, labels

def patch_tSNE():
    storage_dir = '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/supervised/patch_features/'

    train_fname = 'bagnet33_patch_attributes_and_features_train_v4.h5'
    val_fname = 'bagnet33_patch_attributes_and_features_val_v4.h5'
    test_fname = 'bagnet33_patch_attributes_and_features_test_v4.h5'

    store_dir = '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/supervised/tsne_data' 

    qual_thresh = 0.3
    condition_male = """(global_quality < qual_threshold) & (true_label == 1) & (predicted_label == true_label)"""
    condition_female = """(global_quality < qual_threshold) & (true_label == 0) & (predicted_label == true_label)"""

    conditions = [condition_male, condition_female]
    logit_keys = ['male_logit', 'female_logit']
    tsne_paths = [os.path.join(store_dir, 'tsne_batches_'+x+'_v4.npz') for x in ['male', 'female']]

    for i in range(len(conditions)):
        condition = conditions[i]
        logit_key = logit_keys[i]
        tsne_path = tsne_paths[i]
        
        feats_tr, y_tr = get_true_labels_and_features(storage_dir, train_fname, qual_thresh, condition, logit_key)
        feats_val, y_val = get_true_labels_and_features(storage_dir, val_fname, qual_thresh, condition, logit_key)
        feats_test, y_test = get_true_labels_and_features(storage_dir, test_fname, qual_thresh, condition, logit_key)
        
        save_tsne_points([feats_tr, y_tr, feats_val, y_val, feats_test, y_test], savepath=tsne_path)
    
    #print(feats_tr.shape)
    #print(feats_val.shape)
    #print(feats_test.shape)

    #print(len(y_tr))
    #print(len(y_val))
    #print(len(y_test))
    
def patch_tSNE_combined():
    storage_dir = '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/supervised/patch_features/'

    train_fname = 'bagnet33_patch_attributes_and_features_train_v4.h5'
    val_fname = 'bagnet33_patch_attributes_and_features_val_v4.h5'
    test_fname = 'bagnet33_patch_attributes_and_features_test_v4.h5'

    store_dir = '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/supervised/tsne_data' 

    qual_thresh = 0.3
    condition_male = """(global_quality < qual_threshold) & (true_label == 1) & (predicted_label == true_label)"""
    condition_female = """(global_quality < qual_threshold) & (true_label == 0) & (predicted_label == true_label)"""

    conditions = [condition_male, condition_female]
    true_labels = [np.ones, np.zeros]
    logit_key = 'logit'
    tsne_path = os.path.join(store_dir, 'tsne_batches_combined_v4_df_1.npz')

    feats_tr_all = []
    feats_val_all = []
    feats_test_all = []

    y_tr_all = []
    y_val_all = [] 
    y_test_all = []

    for i in range(len(conditions)):
        condition = conditions[i]
        label_fn = true_labels[i]
        
        feats_tr, y_tr = get_true_labels_and_features(storage_dir, train_fname, qual_thresh, condition, logit_key)
        feats_val, y_val = get_true_labels_and_features(storage_dir, val_fname, qual_thresh, condition, logit_key)
        feats_test, y_test = get_true_labels_and_features(storage_dir, test_fname, qual_thresh, condition, logit_key)
        
        #feats_tr = np.hstack((feats_tr, label_fn((feats_tr.shape[0],))))
        #feats_val = np.hstack((feats_val, label_fn((feats_val.shape[0],))))
        #feats_test = np.hstack((feats_test, label_fn((feats_test.shape[0],))))
        
        feats_tr_all.append(feats_tr)
        feats_val_all.append(feats_val)
        feats_test_all.append(feats_test)

        y_tr_all.append(y_tr)
        y_val_all.append(y_val)
        y_test_all.append(y_test)

    feats_tr_combined = np.concatenate(feats_tr_all)
    feats_val_combined = np.concatenate(feats_val_all)
    feats_test_combined = np.concatenate(feats_test_all)

    y_tr_combined = np.concatenate(y_tr_all)
    y_val_combined = np.concatenate(y_val_all)
    y_test_combined = np.concatenate(y_test_all)

    print('Train features shape: {}'.format(feats_tr_combined.shape))
    print('Val features shape: {}'.format(feats_val_combined.shape))
    print('Test features shape: {}'.format(feats_test_combined.shape))

    save_tsne_points([feats_tr_combined, y_tr_combined, feats_val_combined, y_val_combined, feats_test_combined, y_test_combined], alpha=1, savepath=tsne_path)

if __name__ == '__main__':
    patch_tSNE_combined()

import os
from .params import DEFAULT_DNN_TYPE, DEFAULT_INIT, DEFAULT_LABEL_TYPE, DEFAULT_IMG_SIZE 

def get_model_desc_str(dataset, dnn_type=DEFAULT_DNN_TYPE, base_model_init=DEFAULT_INIT,
                   label_type=DEFAULT_LABEL_TYPE):
    #label_type = 'Retina'
    model_desc_str = '%s_%sNet_%s_%s' % (dataset, label_type, dnn_type, base_model_init)
    return model_desc_str

def get_data_file_path():
    return '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/csvfiles/gender_ukb23742_ukb_fundus_EyeQual_Ensemble_ISBI2020_v3_0-5.csv'
    #return '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/qualfiles/ukb23742_ukb_fundus_EyeQual_Ensemble_ISBI2020_v2_0-6.csv'

def get_data_dir(data, img_size=DEFAULT_IMG_SIZE):
    if data == 'kaggle':
        return '/gpfs01/berens/user/iilanchezian/Projects/Confidence-Calibration/Kaggle_DR/kaggle_data/' 
    elif data == 'UKB':
        print(img_size)
        if img_size == 224:
            return '/gpfs01/berens/data/data/biobank/ukb_fundus_224'
        elif img_size == 512:
            return '/gpfs01/berens/data/data/biobank/ukb_fundus_512'
        elif img_size == 587:
            return '/gpfs01/berens/data/data/biobank/ukb_fundus_587'
        else:
            raise NotImplementedError('Image size must be 224, 512 or 587')
    else:
        raise NotImplementedError('Data can be UKB or kaggle')

	
def get_checkpoint_path(dataset, dnn_type=DEFAULT_DNN_TYPE, base_model_init=DEFAULT_INIT,
			label_type=DEFAULT_LABEL_TYPE):
    model_dir = '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/supervised/modelstore/%s/%s/' % (dataset, dnn_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_str = get_model_desc_str(dataset, dnn_type, base_model_init, label_type)
    model_filename = model_str + '_{epoch:03d}_{val_acc:.3f}' + '.hdf5' 
    model_filepath = os.path.join(model_dir, model_filename)
    return model_filepath 


def get_history_path(dataset, dnn_type=DEFAULT_DNN_TYPE, base_model_init=DEFAULT_INIT, label_type=DEFAULT_LABEL_TYPE):
    results_dir = '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/supervised/results/history'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_str = get_model_desc_str(dataset, dnn_type, base_model_init, label_type)
    history_filename = 'History_%s.png' % (model_str)
    history_filepath = os.path.join(results_dir, history_filename)  
    return history_filepath 

def get_roc_path(dataset, dnn_type=DEFAULT_DNN_TYPE, base_model_init=DEFAULT_INIT, label_type=DEFAULT_LABEL_TYPE):
    results_dir = '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/supervised/results/%s/metrics/ROC' % (dataset)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_str = get_model_desc_str(dnn_type, base_model_init, label_type)
    roc_filename = 'ROC_%s.png' % (model_str)
    roc_filepath = os.path.join(results_dir, roc_filename)  
    return roc_filepath 

def get_cm_path(dataset, dnn_type=DEFAULT_DNN_TYPE, base_model_init=DEFAULT_INIT, label_type=DEFAULT_LABEL_TYPE, dataset_type=None):
    results_dir = '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/supervised/results/%s/metrics/ConfMatrix' % (dataset)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_str = get_model_desc_str(dnn_type, base_model_init, label_type)
    if dataset_type:
        cm_filename = 'CM_%s_%s' % (model_str, dataset_type)
    else:
        cm_filename = 'CM_%s.png' % (model_str)
    cm_filepath = os.path.join(results_dir, cm_filename)  
    return cm_filepath 

def get_model_path(dataset, dnn_type=DEFAULT_DNN_TYPE):
    model_dir = '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/supervised/modelstore/%s/%s/' % (dataset, dnn_type)
    #Models without mask
    #model_filename = 'UKB_genderNet_inceptionv3_imagenet_097_0.843.hdf5'
    #model_filename = 'UKB_genderNet_bagnet33_imagenet_093_0.853.hdf5' 

    #Models with mask
    #model_filename = 'UKB_genderNet_bagnet33_imagenet_097_0.837.hdf5' 
    
    #Models with mask and one unit in dense layer 
    #model_filename = 'UKB_genderNet_inceptionv3_imagenet_099_0.832.hdf5'
    #model_filename = 'UKB_genderNet_bagnet9_imagenet_100_0.788.hdf5'
    #model_filename = 'UKB_genderNet_bagnet17_imagenet_100_0.821.hdf5'
    #model_filename = 'UKB_genderNet_bagnet33_imagenet_098_0.835.hdf5'

    #kaggle dr net 
    model_filename = 'kaggle_genderNet_bagnet33_imagenet_072_0.865.hdf5'
    model_filepath = os.path.join(model_dir, model_filename)
    return model_filepath 

def get_tsne_plot_path(dnn_type=DEFAULT_DNN_TYPE, base_model_init=DEFAULT_INIT, label_type=DEFAULT_LABEL_TYPE):
    results_dir = '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/supervised/results/tsne'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_str = get_model_desc_str(dnn_type, base_model_init, label_type)
    tsne_filename = 'tsne_%s.png' % (model_str)
    tsne_filepath = os.path.join(results_dir, tsne_filename)  
    return tsne_filepath 

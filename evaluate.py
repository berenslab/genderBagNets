from keras.models import load_model
import numpy as np 
import os 
import argparse

from utils.paths import get_model_path, get_data_dir, get_model_path, get_roc_path
from utils.params import TRAIN_RATIO, VAL_RATIO, DEFAULT_IMG_SIZE, DEFAULT_KERNEL, DEFAULT_KERNEL_SIZE, DEFAULT_DNN_TYPE, CHANNEL_FIRST, NUM_CLASSES
from utils.data.load_data import get_UKB_dataset, get_kaggle_data_points
from utils.data.preprocessing_utils import load_ukb_fundus_img, load_kaggle_fundus_img, get_preprocess_fn
from utils.plot_utils.plot_fns import plot_roc_auc, plot_confusion_matrix_all


def get_true_labels_and_predictions(data, model, partition, labels, dnn_type=DEFAULT_DNN_TYPE, 
                                    img_size=DEFAULT_IMG_SIZE, data_aug=None, kernel=DEFAULT_KERNEL, 
                                    ksize=DEFAULT_KERNEL_SIZE, mask=True, subtract_bg=True, 
                                    channel_first=CHANNEL_FIRST, num_classes=NUM_CLASSES):
    fundus_directory = get_data_dir(data, img_size=img_size)
    preprocessing_function = get_preprocess_fn(dnn_type)
    
    print('Evaluating the model...')
    
    y_true = []
    y_pred = []
    
    i = 1
    for j, file_ID in enumerate(partition):
        image_file_name = os.path.join(fundus_directory, file_ID)
        if data == 'UKB':
            y_true.append(labels[file_ID])
            img = load_ukb_fundus_img(image_file_name, data_aug=None, mask=mask, 
                                      subtract_bg=subtract_bg, kernel=kernel, 
                                      ksize=ksize, 
                                      preprocessing_function=preprocessing_function,
                                      channel_first=channel_first
                                      )
        elif data == 'kaggle':
            y_true.append(labels[j])
            kernel=None
            ksize=None
            img = load_kaggle_fundus_img(image_file_name+'.jpeg', mask=mask,
                                         kernel=kernel, ksize=ksize,
                                         channel_first=channel_first,
                                         preprocessing_function=preprocessing_function
                                         )
        else:
            img = None
            raise NotImplementedError('Data can be UKB or kaggle')
        softmax_out = np.squeeze(model.predict(img))
        if num_classes > 1:  # p(y=1|x), the probability score from the MALE neuron
            y_pred.append(softmax_out[-1])  # p(y=1|x), the probability score from the MALE neuron
        else:
            y_pred.append(softmax_out)  # p(y=1|x), the probability score from the MALE neuron

        if i % 5000 == 0:
            print('Instance %d reached' % i)
            
        i = i + 1
    
    print('%d instances were evaluated' % (i))

    return np.asarray(y_true), np.asarray(y_pred)

def generate_evaluation_plots(ys, data, dnn_type=DEFAULT_DNN_TYPE):
    y_tr, y_pred_tr, y_val, y_pred_val, y_test, y_pred_test = ys
    roc_path = get_roc_path(data, dnn_type)

    plot_roc_auc(y_tr, y_pred_tr, y_val, y_pred_val, y_test, y_pred_test, dnn_type, savepath=roc_path)
    plot_confusion_matrix_all(data, y_tr, y_pred_tr, y_val, y_pred_val, y_test, y_pred_test, dnn_type)

def evaluate(data, dnn_type=DEFAULT_DNN_TYPE, img_size=DEFAULT_IMG_SIZE, data_aug=None, 
             kernel=DEFAULT_KERNEL, ksize=DEFAULT_KERNEL_SIZE, mask=True,
             subtract_bg=True, channel_first=CHANNEL_FIRST, num_classes=NUM_CLASSES):
    model_path = get_model_path(data, dnn_type)
    model = load_model(model_path)
    if data == 'UKB':
        ukb_data, labels = get_UKB_dataset(train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)
        train_files = ukb_data['train']
        val_files = ukb_data['val']
        test_files = ukb_data['test']
        labels_train = labels_val = labels_test = labels 
    elif data == 'kaggle':
        train_csv = '/gpfs01/berens/user/iilanchezian/Projects/Confidence-Calibration/Kaggle_DR/kaggle_gradable_train.csv'
        val_csv = '/gpfs01/berens/user/iilanchezian/Projects/Confidence-Calibration/Kaggle_DR/kaggle_gradable_val.csv'
        test_csv = '/gpfs01/berens/user/iilanchezian/Projects/Confidence-Calibration/Kaggle_DR/kaggle_gradable_test.csv'

        train_files, labels_train = get_kaggle_data_points(train_csv)
        val_files, labels_val = get_kaggle_data_points(val_csv)
        test_files, labels_test = get_kaggle_data_points(test_csv)
    else:
        raise NotImplementedError('Data can be UKB or kaggle')

    y_tr, y_pred_tr = get_true_labels_and_predictions(data, model, train_files, labels_train, 
                                                      dnn_type=dnn_type, img_size=img_size, 
                                                      kernel=kernel, ksize=ksize, mask=mask,
                                                      subtract_bg=subtract_bg, 
                                                      channel_first=channel_first,
                                                      num_classes=num_classes)
    y_val, y_pred_val = get_true_labels_and_predictions(data, model, val_files, labels_val, 
                                                        dnn_type=dnn_type, img_size=img_size, 
                                                        kernel=kernel, ksize=ksize, mask=mask,
                                                        subtract_bg=subtract_bg,
                                                        channel_first=channel_first,
                                                        num_classes=num_classes)
    y_test, y_pred_test = get_true_labels_and_predictions(data, model,test_files, labels_test, 
                                                          dnn_type=dnn_type, img_size=img_size, 
                                                          kernel=kernel, ksize=ksize, mask=mask, 
                                                          subtract_bg=subtract_bg,
                                                          channel_first=channel_first,
                                                          num_classes=num_classes)

    
    generate_evaluation_plots([y_tr, y_pred_tr, y_val, y_pred_val, y_test, y_pred_test], data, dnn_type=dnn_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA', default=None, help='Specify UKB or kaggle')
    parser.add_argument('--DNN_TYPE', default=DEFAULT_DNN_TYPE, help='Base network type. Default: %s' % (DEFAULT_DNN_TYPE))
    parser.add_argument('--IMG_SIZE', default=DEFAULT_IMG_SIZE, help='Image dimensions. Possible values: 224, 256, 512, 587. \
                                                Default value: %d' % (DEFAULT_IMG_SIZE))
    parser.add_argument('--CHANNEL_FIRST', type=int, default=0, help='If 1 channel corresponds to the first dimension, else last')
    parser.add_argument('--SUBTRACT_BG', type=int, default=0, help='If 1 background subtraction is enabled')
    parser.add_argument('--MASK', type=int, default=0, help='If 1 circular mask function is applied')
    parser.add_argument('--NUM_CLASSES', type=int, default=1, help='If 1 sigmoid else softmax')


    args = parser.parse_args()
    evaluate(args.DATA, args.DNN_TYPE, img_size=int(args.IMG_SIZE), mask=bool(args.MASK), subtract_bg=bool(args.SUBTRACT_BG), channel_first=bool(args.CHANNEL_FIRST), num_classes=args.NUM_CLASSES)

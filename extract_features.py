from keras.models import load_model
import numpy as np 
import os 
import argparse

from utils.paths import get_model_path, get_data_dir, get_tsne_plot_path
from utils.params import TRAIN_RATIO, VAL_RATIO, DEFAULT_IMG_SIZE, DEFAULT_KERNEL, DEFAULT_KERNEL_SIZE, DEFAULT_DNN_TYPE, DEFAULT_LAYER, CHANNEL_FIRST
from utils.data.load_UKB import get_dataset 
from utils.data.UKB_utils import load_ukb_fundus_img
from utils.plot_utils.plot_fns import plot_roc_auc, plot_confusion_matrix_all
from utils.data.preprocess import get_preprocess_fn
from utils.tsne_utils import plot_tsne

from keras.models import Model


def get_true_labels_and_features(model, partition, labels, dnn_type=DEFAULT_DNN_TYPE, 
                                 layer_name=DEFAULT_LAYER, img_size=DEFAULT_IMG_SIZE, 
                                 data_aug=None, kernel=DEFAULT_KERNEL, 
                                 ksize=DEFAULT_KERNEL_SIZE, subtract_bg=True,
                                 channel_first=CHANNEL_FIRST):
  ukb_fundus_directory = get_data_dir(img_size=img_size)
  preprocessing_function = get_preprocess_fn(dnn_type)

  rep_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output) 

  print('Evaluating the model...')

  y_true = []
  features = []

  i = 1
  for file_ID in partition:
    y_true.append(labels[file_ID])
    image_file_name = os.path.join(ukb_fundus_directory, file_ID)
    reps = np.squeeze(rep_model.predict(load_ukb_fundus_img(image_file_name, data_aug=None, 
                                                            subtract_bg=subtract_bg, kernel=kernel, 
                                                            ksize=ksize,
                                                            preprocessing_function=preprocessing_function,
                                                            channel_first=channel_first)
                                                           )
                     )

    features.append(reps)

    if i % 5000 == 0:
      print('Instance %d reached' % i)

    i = i + 1

  print('%d instances were evaluated' % (i))
  features = np.stack(features)
  print(features.shape)

  return np.asarray(y_true), np.asarray(features)


def visualise_features(dnn_type=DEFAULT_DNN_TYPE, layer_name=DEFAULT_LAYER, img_size=DEFAULT_IMG_SIZE, 
                       data_aug=None, kernel=DEFAULT_KERNEL, ksize=DEFAULT_KERNEL_SIZE, 
                       subtract_bg=True, channel_first=CHANNEL_FIRST):
  #model_path = '/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/supervised/modelstore/bestmodels/UKB_RetinaNet_resnet50_imagenet_100_0.778.hdf5'
  model_path = get_model_path(dnn_type)
  model = load_model(model_path)
  print([x.name for x in model.layers])
  data, labels = get_dataset(train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)

  tsne_path = get_tsne_plot_path(dnn_type)

  y_tr, feats_tr = get_true_labels_and_features(model, data['train'], labels, 
                                                dnn_type=dnn_type, layer_name=layer_name,
                                                img_size=img_size, kernel=kernel, 
                                                ksize=ksize, subtract_bg=subtract_bg,
                                                channel_first=channel_first)
  y_val, feats_val = get_true_labels_and_features(model, data['val'], labels, 
                                                  dnn_type=dnn_type, layer_name=layer_name,
                                                  img_size=img_size, kernel=kernel, 
                                                  ksize=ksize, subtract_bg=subtract_bg,
                                                  channel_first=channel_first)
  y_test, feats_test = get_true_labels_and_features(model, data['test'], labels, 
                                                    dnn_type=dnn_type, layer_name=layer_name,
                                                    img_size=img_size, kernel=kernel, 
                                                    ksize=ksize, subtract_bg=subtract_bg,
                                                    channel_first=channel_first)


  plot_tsne([feats_tr, y_tr, feats_val, y_val, feats_test, y_test], savepath=tsne_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--DNN_TYPE', default=DEFAULT_DNN_TYPE, help='Base network type. Default: %s' % (DEFAULT_DNN_TYPE))
  parser.add_argument('--IMG_SIZE', default=DEFAULT_IMG_SIZE, help='Image dimensions. Possible values: 224, 256, 512, 587. \
                                                Default value: %d' % (DEFAULT_IMG_SIZE))
  parser.add_argument('--LAYER_NAME', default=DEFAULT_LAYER, help='Layer of which features are visualized. Default: %s' % (DEFAULT_LAYER))
  parser.add_argument('--CHANNEL_FIRST', type=int, default=0, help='If 0 channel corresponds to the last input dimension else first')
  parser.add_argument('--SUBTRACT_BG', type=int, default=0, help='If 0 background is not subtracted from fundus image else subtracted')
  
  args = parser.parse_args()
  visualise_features(args.DNN_TYPE, args.LAYER_NAME, img_size=int(args.IMG_SIZE), subtract_bg=bool(args.SUBTRACT_BG), channel_first=bool(args.CHANNEL_FIRST))

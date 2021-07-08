import pandas as pd
import numpy as np
import collections

from ..paths import get_data_file_path 
from ..params import DEFAULT_THRESHOLD, DEFAULT_LABEL_TYPE

def get_eid_to_files_mapping(label_type=DEFAULT_LABEL_TYPE):
    labels = {}
    eIDs_to_files = collections.OrderedDict()

    csv_file = get_data_file_path()
    df = pd.read_csv(csv_file)

    for idx, row in df.iterrows():
        if not np.isnan(row[label_type]):
            eid = str(row['filename'])[:7]
            fileid = str(row['filename'])
            labels[fileid] = int(row[label_type])
            if eid in eIDs_to_files:
                eIDs_to_files[eid].append(fileid)
            else:
                eIDs_to_files[eid] = [] 
                eIDs_to_files[eid].append(fileid)

    return eIDs_to_files, labels

def add_files_to_partition(partition, labels, eIDs_to_files, list_eIDs, shuffle_idx, begin, end, 
                           partition_type, oversampling_rate=None, undersampling_rate=None):
    partition[partition_type] = []
    for i, eid in enumerate(np.asarray(list_eIDs)[shuffle_idx[begin:end]].tolist()):
        for j, fileid in enumerate(eIDs_to_files[eid]):
            if partition_type == 'train' and oversampling_rate is not None and labels[fileid] == 1:
                partition[partition_type].extend([fileid] * oversampling_rate)
            else:
                partition[partition_type].append(fileid)

    if undersampling_rate is not None:
        label_vals = list(map(lambda x: labels[x], partition[partition_type]))
        num_healthy = np.sum(np.asarray(label_vals) == 0)
        healthy = list(filter(lambda x: labels[x] == 0, partition[partition_type]))
        healthy_resampled = np.random.choice(healthy, size=int(undersampling_rate * num_healthy))

        partition[partition_type] = list(filter(lambda x: labels[x] == 1, partition[partition_type]))
        partition[partition_type].extend(healthy_resampled)



def partition_data(eIDs_to_files, labels, train_ratio, val_ratio,
                   oversampling_rate=None, undersampling_rate=None):
    partition = {}
    list_eIDs = list(eIDs_to_files.keys())
    shuffle_idx = np.random.RandomState(seed=1828).permutation(len(list_eIDs))

    add_files_to_partition(partition, labels, eIDs_to_files, list_eIDs, shuffle_idx, 0, int(train_ratio*len(list_eIDs)), 'train',
                           oversampling_rate=oversampling_rate, undersampling_rate=undersampling_rate)
    add_files_to_partition(partition, labels, eIDs_to_files, list_eIDs, shuffle_idx, int(train_ratio * len(list_eIDs)), 
                           int((train_ratio + val_ratio) * len(list_eIDs)), 'val')
    add_files_to_partition(partition, labels, eIDs_to_files, list_eIDs, shuffle_idx, int((train_ratio + val_ratio) * len(list_eIDs)), 
                           int(len(list_eIDs)), 'test')

    print_data_report(partition, labels, shuffle_idx, list_eIDs, train_ratio, val_ratio)
    return partition 


def get_categorical_frequencies(partition, labels):
    label_vals = list(map(lambda x: labels[x], partition))
    num_unhealthy = np.sum(np.asarray(label_vals) == 1)
    num_healthy = np.sum(np.asarray(label_vals) == 0)
    return num_healthy, num_unhealthy 

def print_data_report(partition, labels, shuffle_idx, list_eIDs, train_ratio, val_ratio):
    num_healthy_train, num_unhealthy_train = get_categorical_frequencies(partition['train'], labels)
    num_healthy_val, num_unhealthy_val = get_categorical_frequencies(partition['val'], labels)
    num_healthy_test, num_unhealthy_test = get_categorical_frequencies(partition['test'], labels)

    print('The number of training subject (patients) : %g' % 
          len(np.asarray(list_eIDs)[shuffle_idx[:int(train_ratio*len(list_eIDs))]].tolist()))
    print('The number of validation subject (patients) : %g' % 
          len(np.asarray(list_eIDs)[shuffle_idx[int(train_ratio*len(list_eIDs)):int((train_ratio+val_ratio)*len(list_eIDs))]].tolist()))
    print('The number of test subject (patients) : %g' % 
          len(np.asarray(list_eIDs)[shuffle_idx[int((train_ratio+val_ratio)*len(list_eIDs)):]].tolist()))
    print('The total number of subjects (patients) : %g' % (len(np.asarray(list_eIDs)[shuffle_idx[:int(train_ratio*len(list_eIDs))]].tolist()) +
          len(np.asarray(list_eIDs)[shuffle_idx[int(train_ratio*len(list_eIDs)):int((train_ratio+val_ratio)*len(list_eIDs))]].tolist()) +
          len(np.asarray(list_eIDs)[shuffle_idx[int((train_ratio+val_ratio)*len(list_eIDs)):]].tolist()
         )))
    print('------------------------------------------------------------')
    print('The number of training instances : %g' % len(partition['train']))
    print('The number of healthy training instances: %g' % num_healthy_train)
    print('The number of unhealthy training instances: %g' % num_unhealthy_train)
    print('------------------------------------------------------------')
    print('The number of validation instances : %g' % len(partition['val']))
    print('The number of healthy validation instances: %g' % num_healthy_val)
    print('The number of unhealthy validation instances: %g' % num_unhealthy_val)
    print('------------------------------------------------------------')
    print('The number of test instances : %g' % len(partition['test']))
    print('The number of healthy test instances: %g' % num_healthy_test)
    print('The number of unhealthy test instances: %g' % num_unhealthy_test)
    print('------------------------------------------------------------')
    print('The total number of instances : %g' % (len(partition['train'])+ len(partition['val']) + len(partition['test'])))

def get_UKB_dataset(train_ratio, val_ratio, undersampling_rate=None, oversampling_rate=None):
    eIDs_to_files, labels = get_eid_to_files_mapping()
    data = partition_data(eIDs_to_files, labels, train_ratio, val_ratio, 
                          oversampling_rate=oversampling_rate,
                          undersampling_rate=undersampling_rate)
    return data, labels

 
def get_kaggle_data_points(csv_file):
    df = pd.read_csv(csv_file)
    files = df['image'].tolist()
    labels = df['dr'].tolist() #binary TO DO: include multi-class

    return files, labels



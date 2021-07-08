import os
import numpy as np
import pylab as plt
import seaborn as sns
import pickle
import multiprocessing
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
sns.set_style('ticks')

# the path should point to the FIt-SNE directory
import sys; sys.path.append('/gpfs01/berens/user/iilanchezian/Projects/UKbiobank/libs/FIt-SNE')
from fast_tsne import fast_tsne

COLORS=np.array(['#ff0000', '#0000ff'])
LABELS=np.array(['Female', 'Male'])

def find_kNN_idx_per_column(col,k):
    sorted_idx = np.argsort(col)
    return sorted_idx[:k]

def find_kNN_inits_per_column(kNN_idx, Z, dims=2):
    return Z[kNN_idx, :dims]

def reduce_dims(X, map_dims=2):
    #map_dims = 1 # or 2
    variance_to_keep = 0.99
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    print('PCA on test data ...')
    Sigma = np.cov(np.transpose(X))
    U, s, V = np.linalg.svd(Sigma, full_matrices=False)
    sum_s = np.sum(s)
    print('Total components : %g' % len(s))
    for d in range(len(s)):
        var_explained = np.sum(s[:d]) / sum_s
        if var_explained >= variance_to_keep:
            break
    print('%g of variance explained with %d components.' % (var_explained, d))
    D = d
    XD = np.dot(X, U[:,:D])   # np.dot(U, np.diag(s))[:,:D]
    PCAinit = XD[:,:map_dims] / np.std(XD[:,0]) * 0.0001
    return XD, PCAinit, scaler, U, D


def reduce_dims_with_alignment_batches(Xa, Xb, scaler, U, Za, D, map_dims=2, k=10, num_cores=-5, batch_size=1000, multicore_kNN=False):
    Xb = scaler.transform(Xb)
    print('Computing the pairwise distances')

    last = Xb.shape[0] // batch_size
    residue = Xb.shape[0]%batch_size
    kNN_init = []
    for i in np.arange(0, Xb.shape[0], batch_size):
        print('Processing batch no: %d of %d'%(i//batch_size, last))
        if i == last:
            K = pairwise_distances(X=Xa, Y=Xb[i:i+residue,:], metric='euclidean', n_jobs=num_cores)
        else:
            K = pairwise_distances(X=Xa, Y=Xb[i:i+batch_size,:], metric='euclidean', n_jobs=num_cores)
        Ma, Mb = K.shape
        kNN_idx_list_batch = []
        for j in range(Mb):
            idx = np.argsort(K[:,j])
            kNN_idx_list_batch.append(idx[:k])

        for kNN_idx in kNN_idx_list_batch:
            kNNs = Za[kNN_idx, :map_dims]
            kNN_init.append(np.mean(kNNs, axis=0))
    print(len(kNN_init))
    kNN_init = np.reshape(kNN_init, newshape=(Xb.shape[0] ,map_dims))

    XbD = np.dot(Xb, U[:,:D]) 
    
    print(len(kNN_init) == XbD.shape[0])

    return XbD, kNN_init

def reduce_dims_with_alignment(Xa, Xb, scaler, U, Za, D, map_dims=2, k=10, num_cores=10, multicore_kNN=False):
    Xb = scaler.transform(Xb)
    print('Computing the pairwise distances')

    K = pairwise_distances(X=Xa, Y=Xb, metric='euclidean', n_jobs=num_cores)
    Ma, Mb = K.shape
    print('Finding kNNs...')
    kNN_idx_list = []
    if not multicore_kNN:
        for j in range(Mb): # loop over the items to be aligned with the reference map from Xa.
            idx = np.argsort(K[:,j]) # ascending order, so most distant at the end. kNNs are in the front
            kNN_idx_list.append(idx[:k]) # append the kNN indices
    else:
        kNNs_by_idx = Parallel(n_jobs=num_cores)(delayed(find_kNN_idx_per_column)(K[:,j], k) for j in range(Mb))
        for j in range(len(kNNs_by_idx)):
            kNN_idx_list.append(kNNs_by_idx[j])

    XbD = np.dot(Xb, U[:,:D])    # np.dot(U, np.diag(s))[:,:D]

    print('Collecting initialization points based on kNNs')
    kNN_init = []
    if not multicore_kNN:
        for kNN_idx in kNN_idx_list:
            kNNs = Za[kNN_idx,:map_dims]
            kNN_init.append(np.mean(kNNs, axis=0))
        kNN_init = np.reshape(kNN_init, newshape=(len(kNN_idx_list),map_dims))
    else:
        kNN_init = Parallel(n_jobs=num_cores)(delayed(find_kNN_inits_per_column)(kNN_idx_list[j], Za, map_dims) for j in range(len(kNN_idx_list)))
        kNN_init = np.reshape(kNN_init, newshape=(len(kNN_idx_list), k, map_dims))
        kNN_init = np.mean(kNN_init, axis=1)

    return XbD, kNN_init

def plot_tsne_partition(plt, Z, y, idx, title):
    plt.subplot(1,3,idx)
    plt.axis('equal')
    plt.scatter(Z[:,0], Z[:,1], c=COLORS[y], s=2, edgecolors=None, label=LABELS[y])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)


def plot_tsne_partition_logits(plt, Z, y, y_max, y_min,idx, title):
    #norm = plt.Normalize(np.min(y), np.max(y))
    #y = (y - y_min) / (y_max - y_min)
    #y = (2.0 * y) - 1.0
    y = np.array(y)
    y = 1. / (1. + np.exp(-1 * y))
    #y = np.clip(y, 0.0 + 1e-10, 1.0 - 1e-10)
    y = y.tolist()
    plt.subplot(1,3,idx)
    plt.axis('equal')
    plt.scatter(Z[:,0], Z[:,1], c=y, s=1, alpha=0.5, cmap='coolwarm', edgecolors=None)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)


def save_tsne_points(feats_and_labels, map_dims=2, alpha=0.5, savepath=None):
    X_tr, y_tr, X_val, y_val, X_test, y_test = feats_and_labels
    print('df= %.2f'%alpha)

    X_tr_D, tr_init, scaler, U, D = reduce_dims(X_tr)

    #Using perplexity 
    #Z_tr = fast_tsne(X_tr_D, map_dims=map_dims, perplexity=100, learning_rate=10000, initialization=tr_init, seed=42, df=alpha)
    
    #Faster using k and sigma 
    Z_tr = fast_tsne(X_tr_D, map_dims=map_dims, K=15, sigma=1e+6, learning_rate=10000, initialization=tr_init, seed=42, df=alpha)

    #Compute nearest neighbors for alignment in batch mode 
    X_val, val_init = reduce_dims_with_alignment_batches(X_tr, X_val, scaler, U, Z_tr, D)
    X_test, test_init = reduce_dims_with_alignment_batches(X_tr, X_test, scaler, U, Z_tr, D)
    
    #Aligned tSNE using perplexity 
    # Z_val = fast_tsne(X_val, map_dims=map_dims, perplexity=100, learning_rate=10000, initialization=val_init, seed=42, df=alpha)
    # Z_test = fast_tsne(X_test, map_dims=map_dims, perplexity=100, learning_rate=10000, initialization=test_init, seed=42, df=alpha)

    #Aligned tSNE using K and sigma 
    Z_val = fast_tsne(X_val, map_dims=map_dims, K=15, sigma=1e+6, learning_rate=10000, initialization=val_init, seed=42, df=alpha)
    Z_test = fast_tsne(X_test, map_dims=map_dims, K=15, sigma=1e+6, learning_rate=10000, initialization=test_init, seed=42, df=alpha)

    np.savez(savepath, Z_tr=Z_tr, y_tr=np.array(y_tr), Z_val=Z_val, y_val=np.array(y_val), Z_test=Z_test, y_test=np.array(y_test))


def plot_tsne(feats_and_labels, map_dims=2, alpha=0.5, savepath=None):
    X_tr, y_tr, X_val, y_val, X_test, y_test = feats_and_labels

    X_tr_D, tr_init, scaler, U, D = reduce_dims(X_tr)

    #Using perplexity 
    #Z_tr = fast_tsne(X_tr_D, map_dims=map_dims, perplexity=100, learning_rate=10000, initialization=tr_init, seed=42, df=alpha)
    
    #Faster using k and sigma 
    #How to choose sigma value 
    Z_tr = fast_tsne(X_tr_D, map_dims=map_dims, K=15, sigma=1e+6, learning_rate=10000, initialization=tr_init, seed=42, df=alpha)

    #Compute nearest neighbors for alignment in batch mode 
    X_val, val_init = reduce_dims_with_alignment_batches(X_tr, X_val, scaler, U, Z_tr, D)
    X_test, test_init = reduce_dims_with_alignment_batches(X_tr, X_test, scaler, U, Z_tr, D)
    
    #Aligned tSNE using perplexity 
    # Z_val = fast_tsne(X_val, map_dims=map_dims, perplexity=100, learning_rate=10000, initialization=val_init, seed=42, df=alpha)
    # Z_test = fast_tsne(X_test, map_dims=map_dims, perplexity=100, learning_rate=10000, initialization=test_init, seed=42, df=alpha)

    #Aligned tSNE using K and sigma 
    Z_val = fast_tsne(X_val, map_dims=map_dims, K=15, sigma=1e+6, learning_rate=10000, initialization=val_init, seed=42, df=alpha)
    Z_test = fast_tsne(X_test, map_dims=map_dims, K=15, sigma=1e+6, learning_rate=10000, initialization=test_init, seed=42, df=alpha)

    np.savez('tsne_batches_female_v2', Z_tr=Z_tr, y_tr=np.array(y_tr), Z_val=Z_val, y_val=np.array(y_val), Z_test=Z_test, y_test=np.array(y_test))

    plt.figure(figsize=(10,10))

    #y_max = np.max([np.max(y_tr), np.max(y_val), np.max(y_test)])
    #y_min = np.min([np.min(y_tr), np.min(y_val), np.min(y_test)])

    #norm = plt.Normalize(y_max, y_min)
    y_max = np.max(y_tr)
    y_min = np.min(y_tr)

    plot_tsne_partition_logits(plt, Z_tr, y_tr, y_max, y_min, 1, 'Training set')
    plot_tsne_partition_logits(plt, Z_val, y_val, y_max, y_min, 2, 'Validation set')
    plot_tsne_partition_logits(plt, Z_test, y_test, y_max, y_min, 3, 'Test set')

    plt.colorbar()
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)
        return None
    else:
        plt.show()
    



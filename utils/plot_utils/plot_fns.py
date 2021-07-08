from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np 

from ..paths import get_cm_path

def plot_history(history, savepath=None):
    fig = plt.figure(figsize=(16,21))
    titles = ['Fine tuning FC\n', 'Fine tuning full model','Fine tuning full model']

    i = 1
    for hist, title in zip(history, titles):
        # summarize history for loss
        fig.add_subplot(3,2,i)
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title(title + ' loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        i = i + 1

        fig.add_subplot(3,2,i)
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.title(title + ' accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        i = i + 1

    if savepath is not None:
        plt.savefig(savepath)
        return None 
    else:
        plt.show()

def plot_roc_auc(y_tr, y_pred_tr, y_val, y_pred_val, y_test, y_pred_test, dnn_type, savepath=None):
    fpr_tr, tpr_tr, _ = roc_curve(y_tr, y_pred_tr)
    roc_auc_tr = auc(fpr_tr, tpr_tr)

    fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_val)
    roc_auc_val = auc(fpr_val, tpr_val)

    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test)
    roc_auc_test = auc(fpr_test, tpr_test)

    plt.figure()
    lw = 2
    plt.plot(fpr_tr, tpr_tr, color='darkblue',
             lw=lw, label='Training set (AUC = %0.2f)' % roc_auc_tr)
    plt.plot(fpr_val, tpr_val, color='darkorange',
             lw=lw, label='Validation set (AUC = %0.2f)' % roc_auc_val)
    plt.plot(fpr_test, tpr_test, color='darkgreen',
             lw=lw, label='Test set (AUC = %0.2f)' % roc_auc_test)
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('%s'%(dnn_type.capitalize()), fontsize=14)
    plt.legend(loc="lower right")

    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()

def plot_confusion_matrix(y, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, savepath=None):
    cm = confusion_matrix(y, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()

def plot_confusion_matrix_all(data, y_tr, y_pred_tr, y_val, y_pred_val, y_test, y_pred_test, dnn_type):
    train_cm_path = get_cm_path(data, dataset_type='train', dnn_type=dnn_type)
    plot_confusion_matrix(y_tr, np.int32(np.greater_equal(y_pred_tr, 0.5)), classes=np.unique(y_tr),
                          normalize=False, title='%s: Confusion matrix (train)'%(dnn_type), cmap=plt.cm.Blues,
                          savepath=train_cm_path)

    val_cm_path = get_cm_path(data, dataset_type='val', dnn_type=dnn_type)
    plot_confusion_matrix(y_val, np.int32(np.greater_equal(y_pred_val, 0.5)), classes=np.unique(y_val),
                          normalize=False, title='%s: Confusion matrix (val)'%dnn_type, cmap=plt.cm.Blues,
                          savepath=val_cm_path)

    test_cm_path = get_cm_path(data, dataset_type='test', dnn_type=dnn_type)
    plot_confusion_matrix(y_test, np.int32(np.greater_equal(y_pred_test, 0.5)), classes=np.unique(y_test),
                          normalize=False, title='%s: Confusion matrix (test)'%dnn_type, cmap=plt.cm.Blues,
                          savepath=test_cm_path)



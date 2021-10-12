import numpy as np
from PIL import Image
import seaborn as sn
from skimage.transform import resize
import os
import pandas as pd
import matplotlib.pyplot as plt

label_mapping = {0: 0,
                 1: 0,
                 3: 1,
                 4: 2,
                 5: 3,
                 6: 4,
                 7: 5,
                 8: 6,
                 9: 7,
                 10: 8,
                 12: 9,
                 15: 10,
                 17: 11,
                 18: 12,
                 19: 13,
                 23: 14,
                 27: 15,
                 29: 1,
                 30: 1,
                 31: 16,
                 32: 4,
                 33: 17,
                 34: 18}

classname_list = ["void", "grass", "tree", "pole", "water", "sky", "vehicle", "object", "asphalt",
                  "building", "log", "person", "fence", "bush", "concrete", "barrier", "puddle", "mud", "rubble"]


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    seg_pred = pred.flatten().astype('int32')
    seg_gt = label.flatten().astype('int32')
    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]
    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))
    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    
    return confusion_matrix


def convert_label(label, label_mapping, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label


def plot_confusion_matrix(cm, classname_list):
    cm_sum = cm.sum(axis=1)
    cm_sum[cm_sum == 0] = 0.1

    cmn = cm/cm_sum[:, np.newaxis]
    #cmn = cm
    df_cm = pd.DataFrame(cmn, index=classname_list,
                         columns=classname_list)
    fig = plt.figure(figsize=(20, 14))
    sn.heatmap(df_cm, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


root = "/mnt/data_disk/datasets/rellis-3d/Rellis-3D"
list_path = "/mnt/data_disk/datasets/rellis-3d/test.lst"
num_class = 19
img_list = [line.strip().split()[1] for line in open(list_path)]
confusion_matrix = np.zeros((num_class,num_class)).astype(np.float64)
for index, img_path in enumerate(img_list[:]):
    label_path = os.path.join(root,"rellis",img_path)
    pred_path = os.path.join(root,"hrnet",img_path)
    label = Image.open(label_path)
    label = np.array(label)
    label = convert_label(label, label_mapping)
    label_shape = label.shape
    pred = Image.open(pred_path)
    if label_shape[0] != pred.size[0] or label_shape[1] != pred.size[1]:
        pred = pred.resize((label_shape[1],label_shape[0]),Image.NEAREST)
    pred = np.array(pred)[:,:,0]
    pred = convert_label(pred, label_mapping)
    confusion_matrix =confusion_matrix + get_confusion_matrix(label,pred,label.shape,num_class,0)
    if index % 100 == 0:
        print('processing: %d images' % index)
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        print('mIoU: %.4f' % (mean_IoU))
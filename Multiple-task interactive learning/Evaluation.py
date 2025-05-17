import numpy as np
import os
import pandas as pd
from helper_ply import read_ply, write_ply
from sklearn.metrics import confusion_matrix,accuracy_score, average_precision_score
ROOM_PATH_LIST = [line.rstrip() for line in open('/media/sever/data1/zdd/pointnet2/Tongji/meta/shengmingyixue_data_label.txt')]
num_room = len(ROOM_PATH_LIST)

gt_classes = [0 for _ in range(8)]
positive_classes = [0 for _ in range(8)]
true_positive_classes = [0 for _ in range(8)]

oalist=0
for i in range(num_room):
    pppre='/media/sever/data1/shoujun/Tongji/test/1220083/val_predictions/'+os.path.basename(ROOM_PATH_LIST[i])[:-4] + '.txt'
    gggt='/media/sever/data1/shoujun/Tongji/Tongji_dataset/original_ply/'+os.path.basename(ROOM_PATH_LIST[i])[:-4] + '.ply'
    print(pppre)
    print(gggt)
    pred_label = pd.read_csv(pppre, header=None, delim_whitespace=True).values

    data = read_ply(gggt)
            
    gt_label= data['class']

    conf_matrix = confusion_matrix(gt_label, pred_label, np.arange(0, 6, 1))

    oalist = oalist+conf_matrix

    print(gt_label.shape)


def compute_OA(conf_mat):
    recall1 = []
    precision1 = []

    sss = 0
    f1 = 0
    conf_mat1 = conf_mat
    for i in range(6):
        recall1.append(conf_mat1[i][i] / float(conf_mat1[:, i].sum()))
    for i in range(6):
        precision1.append(conf_mat1[i][i] / float(conf_mat1[i, :].sum()))

    for i in range(6):
        sss = sss + conf_mat1[i][i]
    oa = sss / float(conf_mat1.sum())
    iou=[]
    for i in range(6):
        iou.append(conf_mat1[i][i] / float(conf_mat1[i, :].sum() + conf_mat1[:, i].sum() - conf_mat1[i][i]))
    return recall1, precision1, oa, iou
recall1, precision1, oa, iou=compute_OA(oalist)
print('Overall accuracy: {0}'.format(oa))
print('mean accuracy: {0}'.format(sum(precision1)/8.0))
print('mean iou: {0}'.format(sum(iou) / float(8)))
print(iou)

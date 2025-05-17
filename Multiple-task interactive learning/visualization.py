import numpy as np
import os
from os.path import join, exists
import pandas as pd
from helper_ply import write_ply,read_ply
ROOM_PATH_LIST = [line.rstrip() for line in open('/media/sever/data1/shoujun/building_data_label.txt')]
num_room = len(ROOM_PATH_LIST)

label_to_names = { 0: 'building',
                   1: 'Grassland',
                   2: 'Ground',
                   3: 'Tree'}
g_class2color = {0:[255,0,0],
                 1:[0,255,0],
                 2:[0,0,255],
                 3:[0,102,0]}


for i in range(num_room):
    file=os.path.basename(ROOM_PATH_LIST[i])[:-4]
    visualization_path = '/media/sever/data1/shoujun/Tongji/test/Log_2021-01-07_11-11-48/visualization'
    pppre='/media/sever/data1/shoujun/Tongji/test/Log_2021-01-07_11-11-48/val_predictions/'+os.path.basename(ROOM_PATH_LIST[i])[:-4] + '.txt'
    gggt='/media/sever/data1/shoujun/Tongji/Tongji_dataset/original_ply/'+os.path.basename(ROOM_PATH_LIST[i])[:-4] + '.ply'
    print(pppre)
    print(gggt)
    pred_label = pd.read_csv(pppre, header=None, delim_whitespace=True).values
    pred_label =pred_label.astype(np.uint8)
    
    data = read_ply(gggt)
    xyz = np.vstack((data['x'], data['y'], data['z'])).T
    gt_label= data['class'].astype(np.uint8)
    
    
    pre_list = []
    gt_list = []
    for i in range(len(gt_label)):
        pre_list.append(g_class2color[pred_label[i,0]])
        gt_list.append(g_class2color[gt_label[i]])
    pre_colour = np.array(pre_list)
    pre_colour = pre_colour.astype(np.uint8)

    gt_colour = np.array(gt_list)
    gt_colour = gt_colour.astype(np.uint8)
    pre_ply_path = join(visualization_path, file + '_pred_ours.ply')

    gt_ply_path = join(visualization_path, file + '_gt_souyou.ply')
    write_ply(pre_ply_path, (xyz, pre_colour), ['x', 'y', 'z', 'red', 'green', 'blue'])
    write_ply(gt_ply_path, (xyz, gt_colour), ['x', 'y', 'z', 'red', 'green', 'blue'])
    print(pre_ply_path)





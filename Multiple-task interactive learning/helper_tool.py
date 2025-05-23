from open3d import linux as open3d
from os.path import join
import numpy as np
import colorsys, random, os, sys
import pandas as pd
from scipy.spatial import cKDTree
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors


class ConfigTONGJI:
    k_n = 16  # KNN

    num_points = 4096*10 #40960    #65536  # Number of input points
    num_classes = 8  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 3  # yuan ben shi 5 batch_size during training
    val_batch_size = 16  # batch_size during validation and test
    train_steps = 600  # Number of steps per epochs
    val_steps = 225 #yuanlai225 # Number of validation steps per epoch

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_occlusion = 'none'
    augment_color = 0.8



class DataProcessing:

    @staticmethod
    def get_file_list(dataset_path, test_scan_num):
        seq_list = np.sort(os.listdir(dataset_path))

        train_file_list = []
        test_file_list = []
        val_file_list = []
        for seq_id in seq_list:
            seq_path = join(dataset_path, seq_id)
            pc_path = join(seq_path, 'velodyne')
            if seq_id == '08':
                val_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif int(seq_id) >= 11 and seq_id == test_scan_num:
                test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                train_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        train_file_list = np.concatenate(train_file_list, axis=0)
        val_file_list = np.concatenate(val_file_list, axis=0)
        test_file_list = np.concatenate(test_file_list, axis=0)
        return train_file_list, val_file_list, test_file_list

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU


    @staticmethod
    def norm_feature(fea):
        '''normalize the frature (n,1)'''
        max_fea = np.max(fea)
        min_fea = np.min(fea)
        return (fea - min_fea) / (max_fea - min_fea)

    @staticmethod
    def local_feature_extraction(point_clouds, nn_size=10):

        num_point = np.shape(point_clouds)[0]
        xyz_data = point_clouds[:, 0:3]

        tree1 = cKDTree(xyz_data)
        dist, idx = tree1.query(xyz_data, k=nn_size + 1)

        neighborhoods = xyz_data[idx]
        Ex = np.average(neighborhoods, axis=1)
        Ex = np.reshape(np.tile(Ex, [neighborhoods.shape[1]]), neighborhoods.shape)
        P = neighborhoods - Ex
        cov_ = np.matmul(P.transpose((0, 2, 1)), P) / (neighborhoods.shape[1] - 1)
        eigen_, vec_ = np.linalg.eig(cov_)

        epsilon_to_add = 1e-8
        EVs = eigen_
        EVs = np.sort(EVs, axis=1)
        np.putmask(EVs, EVs <= 0, epsilon_to_add)

        # EVs=np.zeros((num_point,3)
        # 2D cov and eigen
        cov2d_ = cov_[:, :2, :2]
        EVs_2d, _ = np.linalg.eig(cov2d_)
        EVs_2d = np.sort(EVs_2d, axis=1)
        np.putmask(EVs_2d, EVs_2d <= 0, epsilon_to_add)
        sum_EVs_2d = np.sum(EVs_2d, axis=1)
        sum_EVs_2d = sum_EVs_2d.reshape(num_point, 1)
        # print(sum_EVs_2d.shape)

        # list_e=[]
        listheight_difference = []
        listheight_variance = []

        # m=nn_size+1
        j = 0
        # xyz_data[idx]
        for j in range(num_point):
            P = neighborhoods[j, :, :]

            listheight_difference.append(np.max(P[:, 2]) - np.min(P[:, 2]))
            listheight_variance.append(np.std(P[:, 2]))

        height_difference = np.array(listheight_difference)
        height_variance = np.array(listheight_variance)

        sum_EVs = np.sum(EVs, axis=1)  # EVs[j,0]+EVs[j,1]+EVs[j,2]    #(num_point,1)
        sum_EVs = sum_EVs.reshape(num_point, 1)

        EVs[:, 0] = EVs[:, 0] / sum_EVs[:, 0]
        EVs[:, 1] = EVs[:, 1] / sum_EVs[:, 0]
        EVs[:, 2] = EVs[:, 2] / sum_EVs[:, 0]

        linearity = (EVs[:, 2] - EVs[:, 1]) / EVs[:, 2]  # (max-zhong)/max    #(num_point,)
        planarity = (EVs[:, 1] - EVs[:, 0]) / EVs[:, 2]  # (num_point,)
        scattering = EVs[:, 0] / EVs[:, 2]
        omnivariance = np.cbrt(EVs[:, 0] * EVs[:, 1] * EVs[:, 2])  # (num_point,)
        anisotropy = (EVs[:, 2] - EVs[:, 0]) / EVs[:, 2]
        eigenentropy = -(EVs[:, 2] * np.log(EVs[:, 2]) + EVs[:, 1] * np.log(EVs[:, 1]) + EVs[:, 0] * np.log(EVs[:, 0]))

        change_of_curvature = EVs[:, 0] / (EVs[:, 0] + EVs[:, 1] + EVs[:, 2])

        linearity = linearity.reshape(num_point, 1)  # (max-zhong)/max
        planarity = planarity.reshape(num_point, 1)
        scattering = scattering.reshape(num_point, 1)
        omnivariance = omnivariance.reshape(num_point, 1)
        anisotropy = anisotropy.reshape(num_point, 1)  # 数值都接近1
        eigenentropy = eigenentropy.reshape(num_point, 1)
        change_of_curvature = change_of_curvature.reshape(num_point, 1)

        density = (nn_size + 1) / (4 / 3 * np.pi * np.power(dist[:, nn_size], 3))
        density = density.reshape(num_point, 1)
        norm_density = DataProcessing.norm_feature(density)  # guiyihua mi du

        height_difference = height_difference.reshape(num_point, 1)
        normheight_difference = DataProcessing.norm_feature(height_difference)

        height_variance = height_variance.reshape(num_point, 1)
        normheight_variance = DataProcessing.norm_feature(height_variance)

        point_cloud1 = np.concatenate((linearity, planarity, scattering,
                                       omnivariance, anisotropy, eigenentropy, change_of_curvature,
                                       sum_EVs, norm_density, normheight_difference,
                                       normheight_variance, sum_EVs_2d), axis=1)
        return point_cloud1
class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        pc = open3d.PointCloud()
        pc.points = open3d.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            open3d.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = open3d.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = open3d.Vector3dVector(pc_xyzrgb[:, 3:6])
        open3d.draw_geometries([pc])
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, fix_color_num=None):
        if fix_color_num is not None:
            ins_colors = Plot.random_colors(fix_color_num + 1, seed=2)
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)  # cls 14

        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if fix_color_num is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0]);
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1]);
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2]);
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins)
        return Y_semins

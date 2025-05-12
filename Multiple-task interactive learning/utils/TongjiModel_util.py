""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util
import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors

def knn_query(k, support_pts, query_pts):
    """
    :param support_pts: points you have, B*N1*3
    :param query_pts: points you want to know the neighbour index, B*N2*3
    :param k: Number of neighbours in knn search
    :return: neighbor_idx: neighboring points indexes, B*N2*k
    """
    neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
    return neighbor_idx.astype(np.int32)

def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    select_idx=farthest_point_sample(npoint, xyz)
    new_xyz = gather_point(xyz, select_idx) # (batch_size, npoint, 3)
    
    if knn:
        #_,idx = knn_point(nsample, xyz, new_xyz)
        idx = tf.py_func(knn_query, [nsample, xyz, new_xyz], tf.int32)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
        
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz
def sample_point(npoint, nsample, xyz, knn=True):
    '''
    Input:
        npoint: int32 下采样点数
        radius: float32
        nsample: int32 K个临近点
        xyz: (batch_size, ndataset, 3) TF tensor 
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
            select_idx: (batch_size, npoint)
    '''
    select_idx=farthest_point_sample(npoint, xyz)
    new_xyz = gather_point(xyz, select_idx) # (batch_size, npoint, 3)

    #new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    if knn:
        #_,idx = knn_point(nsample, xyz, new_xyz)
        idx = tf.py_func(knn_query, [nsample, xyz, new_xyz], tf.int32)
        
    neighbor_xyz = group_point(xyz, idx)   # (batch_size, N, k, 3)
    xyz_tile = tf.tile(tf.expand_dims(new_xyz, axis=2), [1, 1, nsample, 1])
    relative_xyz = xyz_tile - neighbor_xyz
    relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keep_dims=True))
    
    delta_xyz=relative_xyz
    relative_xyz = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1) 
    
    
    #grouped_features = group_point(tf.squeeze(features, axis=2), idx) # (batch_size, npoint, nsample, channel)
    
    return new_xyz,relative_xyz,idx,delta_xyz,select_idx

def weight_net_hidden(xyz, hidden_units, scope, is_training, bn_decay=None, weight_decay = None, activation_fn=tf.nn.relu):

    with tf.variable_scope(scope) as sc:
        net = xyz
        for i, num_hidden_units in enumerate(hidden_units):
            net = tf_util.conv2d(net, num_hidden_units, [1, 1],
                                padding = 'VALID', stride=[1, 1],
                                bn = True, is_training = is_training, activation_fn=activation_fn,
                                scope = 'wconv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

            #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='wconv_dp%d'%(i))
    return net
def sample_point_feature(npoint, nsample, xyz,feature):
    '''
    Input:
        npoint: int32 下采样点数    nsample: int32 K个临近点
        xyz: (batch_size, ndataset, 3) TF tensor 
        feature: (batch_size, ndataset, C) TF tensor 
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_feature: (batch_size, npoint, channel) TF tensor
        
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        neighbor_feature  (batch_size, npoint, nsample,channel) TF tensor
    '''
    select_idx=farthest_point_sample(npoint, xyz)   # batch_size * npoint 
    new_xyz = gather_point(xyz, select_idx)        # (batch_size, npoint, 3)
    
    select_idx1=tf.expand_dims(select_idx, axis=2)  # batch_size * npoint*1 
    new_feature= group_point(feature, select_idx1) # (batch_size, npoint, 1, C)
    new_feature=tf.squeeze(new_feature, [2])     # (batch_size, npoint,  C)
    
    return new_xyz,new_feature
def group_point_feature(npoint, nsample, xyz,feature,new_xyz, new_feature,knn=True):
    '''
    Input:
        npoint: int32 下采样点数    nsample: int32 K个临近点
        xyz: (batch_size, ndataset, 3) TF tensor 
        feature: (batch_size, ndataset, C) TF tensor 
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_feature: (batch_size, npoint, channel) TF tensor
        
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        neighbor_feature  (batch_size, npoint, nsample,channel) TF tensor
    '''
   
    if knn:
        #_,idx = knn_point(nsample, xyz, new_xyz)
        idx = tf.py_func(knn_query, [nsample, xyz, new_xyz], tf.int32)
        
    neighbor_xyz = group_point(xyz, idx)   # (batch_size, N, k, 3)
    xyz_tile = tf.tile(tf.expand_dims(new_xyz, axis=2), [1, 1, nsample, 1])
    relative_xyz = xyz_tile - neighbor_xyz
    relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keep_dims=True))
    
    
    relative_xyz = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1) 
    
    neighbor_feature=group_point(feature, idx) 
    
    
    
    return relative_xyz,idx,neighbor_xyz,neighbor_feature
def pointresnet_module(xyz, features, npoint, nsample, d_out,is_training, bn_decay, scope, bn=True, knn=True, use_nchw=False):
    '''
    xyz:points(B,N,3)
    features: (B,N,1,C)
    k: Nearest K neighbors  nsample
    
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        
        #new_xyz,relative_xyz,idx,delta_xyz,select_idx= sample_point(npoint, nsample, xyz,knn)
        new_xyz,new_feature=sample_point_feature(npoint, nsample, xyz,features)
        relative_xyz,idx,neighbor_xyz,neighbor_feature=group_point_feature(npoint, nsample, xyz,features,new_xyz, new_feature,knn)
        
        concat_feature=tf.concat([new_xyz, new_feature], axis=-1)       # (batch_size, npoint, 3+channel)
        concat_feature=tf.expand_dims(concat_feature, axis=2)           # (batch_size, npoint, 1,3+channel)
        
        
        concat_feature =tf_util.conv2d(concat_feature, d_out, [1,1],                 
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(0), bn_decay=bn_decay,
                                        data_format=data_format)
        new_features =global_context_module(concat_feature, is_training, bn_decay, bn,scope,data_format)
        ''' KNN'''
        f_neighbours1 =tf_util.conv2d( neighbor_feature, d_out//2, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(1), bn_decay=bn_decay,
                                        data_format=data_format) 
        
          # (batch_size, npoint, nsample, channel)
        f_xyz =tf_util.conv2d(relative_xyz, d_out//2, [1,1],                
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(2), bn_decay=bn_decay,
                                        data_format=data_format)    # # (batch_size, N, k, 3)
        
        f_concat = tf.concat([ f_neighbours1, f_xyz], axis=-1)
        
        
        f_concat =tf_util.conv2d( f_concat, d_out//2, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(3), bn_decay=bn_decay,
                                        data_format=data_format)
        
        f_concat =tf_util.conv2d( f_concat, d_out, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(31), bn_decay=bn_decay,
                                        data_format=data_format)
        
        shortcut =tf_util.conv2d(tf.concat([neighbor_feature, relative_xyz], axis=-1), d_out, [1,1],                 
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(4), bn_decay=bn_decay,
                                        data_format=data_format)
        
        weight = weight_net_hidden(neighbor_xyz, [32], scope = 'weight_net', is_training=is_training, bn_decay = bn_decay)
        f_concat = tf.transpose(f_concat, [0, 1, 3, 2])
        f_concat = tf.matmul(f_concat, weight)      # (batch_size, npoint, k, 32)
        f_concat = tf.reduce_max(f_concat, axis=[2], keep_dims=True, name='maxpool')  # (batch_size, npoint, 1, channel)
        
        shortcut1 =tf_util.conv2d(shortcut, d_out, [1,1],                 
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(41), bn_decay=bn_decay,
                                        data_format=data_format)
        
        shortcut2 = tf.reduce_max(shortcut1, axis=[2], keep_dims=True, name='maxpool')  # (batch_size, npoint, 1, channel)
        
        f_concat =tf_util.conv2d(tf.concat([f_concat, shortcut2], axis=-1), d_out, [1,1],                 
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(42), bn_decay=bn_decay,
                                        data_format=data_format)
        
        new_features =tf_util.conv2d(new_features+f_concat, d_out, [1,1],               
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(5), bn_decay=bn_decay,
                                        data_format=data_format)
        
        new_features =tf_util.conv2d(new_features, d_out, [1,1],               
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(51), bn_decay=bn_decay,
                                        data_format=data_format)
       
        
        #new_features=tf.nn.leaky_relu(new_features)
         
     
        new_features= tf.squeeze(new_features, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_features, idx
'''
Input:
    center_xyz: sampled points position data [B, npoint, C]
    center_feature: centered point feature [B, npoint, D]
    grouped_xyz: group xyz data [B, npoint, nsample, C]
    grouped_feature: sampled points feature [B, npoint, nsample, D]
Return:
    graph_pooling: results of graph pooling [B, npoint, D]

B, npoint, C = center_xyz.size()
_, _, nsample, D = grouped_feature.size()
delta_p = center_xyz.view(B, npoint, 1, C).expand(B, npoint, nsample, C) - grouped_xyz # [B, npoint, nsample, C]
delta_h = center_feature.view(B, npoint, 1, D).expand(B, npoint, nsample, D) - grouped_feature # [B, npoint, nsample, D]
delta_p_concat_h = torch.cat([delta_p,delta_h],dim = -1) # [B, npoint, nsample, C+D]
e = self.leakyrelu(torch.matmul(delta_p_concat_h, self.a)) # [B, npoint, nsample,D]
attention = F.softmax(e, dim=2) # [B, npoint, nsample,D]
attention = F.dropout(attention, self.dropout, training=self.training)
graph_pooling = torch.sum(torch.mul(attention, grouped_feature),dim = 2) # [B, npoint, D]
return 
'''
def global_context_module(feature, is_training, bn_decay, bn,scope,data_format):
    '''
        Input:
            feature:  [B, npoint,1, C]
            delta_feature: centered point feature # [B, npoint, nsample, D]
            mix_feature: sampled points feature [B, npoint, nsample, D]
        Return:
            att_frature: results of graph pooling [B, npoint, nsample, D]
    '''
    with tf.variable_scope(scope) as sc:
         Batch_size = tf.shape(feature)[0]
         Num_points = tf.shape(feature)[1]
         Num_k = tf.shape(feature)[2]
         Channel=feature.get_shape()[3].value
         context_mask = tf_util.conv2d(feature, 1, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='global_context1', bn_decay=bn_decay,
                                            data_format=data_format)   # [N, H, W,1]
         context_mask = tf.reshape(context_mask,[Batch_size,  -1, 1])# [N, H*W, 1]
         context_mask=tf.transpose(context_mask,perm=[0,2,1])# [N, 1, H*W]
         context_mask = tf.nn.softmax(context_mask)# [N, 1, H*W]
         input_x = tf.reshape(feature, [Batch_size,  -1, Channel])# [N,H*W,C]
         context_mask=tf.matmul(context_mask,input_x)# [N, 1, H*W] x [N,H*W,C] =[N,1,C]
         context_mask= tf.reshape( context_mask, shape=[Batch_size, 1, 1, Channel])
         #context_mask=tf.expand_dims(context_mask,axis=1)#[N,1,1,C]
         
         context_mask = tf_util.conv2d(context_mask, Channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='global_context2', bn_decay=bn_decay,
                                            data_format=data_format)   # [N, 1, 1,C]
         
         context_mask=tf.contrib.layers.layer_norm(context_mask,center=True, scale=True,scope=scope)
         
         context_mask=tf.nn.relu(context_mask)
         context_mask = tf_util.conv2d(context_mask, Channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='global_context3', bn_decay=bn_decay,
                                            data_format=data_format)   # [N, 1, 1,C]
         #context_transform = tf.sigmoid(context_mask)
         feature = feature + context_mask
    return feature
def attention_pooling_module(feature,d_out,delta_xyz,delta_feature,is_training, scope,bn,bn_decay,data_format,i):
    '''
        Input:
            delta_xyz: sampled points position data # [B, npoint, nsample, C]
            delta_feature: centered point feature # [B, npoint, nsample, D]
            mix_feature: sampled points feature [B, npoint, nsample, D]
        Return:
            att_frature: results of graph pooling [B, npoint, nsample, D]
    '''
    with tf.variable_scope(scope) as sc:
        batch_size1 = tf.shape(feature)[0]
        num_points1 = tf.shape(feature)[1]
        num_neigh1 = tf.shape(feature)[2]
        d1 = feature.get_shape()[3].value
        f_concatcddelta = tf.concat([delta_xyz,delta_feature], axis=-1)  #[B, npoint, nsample, C+D]
        #
        line_activation = tf.layers.dense( f_concatcddelta, d1, activation=None, use_bias=False, name='line_%d'%(i))
        line_activation=tf.nn.leaky_relu(line_activation)  #[B, npoint, nsample, D]
        
        f_reshaped_line = tf.reshape(line_activation, shape=[-1, num_neigh1, d1])  #[B*npoint, nsample, D]
        att_maps = tf.nn.softmax(f_reshaped_line)
        att_maps=tf_util.dropout(att_maps, keep_prob=0.6, is_training=is_training, scope=scope)
        
        f_reshaped_feature = tf.reshape(feature, shape=[-1, num_neigh1, d1])  #[B*npoint, nsample, D]
        att_feature =f_reshaped_feature*att_maps    #[B*npoint, nsample,D]
        
        att_feature = tf.reduce_sum(att_feature, axis=1)
        att_feature = tf.reshape(att_feature, [batch_size1, num_points1, 1, d1])
        
        att_feature = tf_util.conv2d(att_feature, d_out, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='attention_pooling_module%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format) 
        
    return att_feature   
'''
 context=conv(context,c,1)#[N,1,1,C]
 
 n,h,w,c=feature.get_shape().as_list()
 context_mask = conv(x,1,1)# [N, H, W,1]
 context_mask = tf.reshape(context_mask,shape=tf.convert_to_tensor([tf.shape(x)[0], -1, 1]))# [N, H*W, 1]
 context_mask=tf.transpose(context_mask,perm=[0,2,1])# [N, 1, H*W]
 context_mask = tf.nn.softmax(context_mask,axis=2)# [N, 1, H*W]

 input_x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], -1,c]))# [N,H*W,C]

 context=tf.matmul(context_mask,input_x)# [N, 1, H*W] x [N,H*W,C] =[N,1,C]
 context=tf.expand_dims(context,axis=1)#[N,1,1,C]
 context=conv(context,squeeze_depth,1)
 context=slim.layer_norm(context)
 context=tf.nn.relu(context)
 context=conv(context,c,1)#[N,1,1,C]
'''
def att_block(feature,delta_xyz,delta_feature,is_training, scope,i):
    '''
        Input:
            delta_xyz: sampled points position data # [B, npoint, nsample, C]
            delta_feature: centered point feature # [B, npoint, nsample, D]
            mix_feature: sampled points feature [B, npoint, nsample, D]
        Return:
            att_frature: results of graph pooling [B, npoint, nsample, D]
    '''
    with tf.variable_scope(scope) as sc:
        batch_size1 = tf.shape(feature)[0]
        num_points1 = tf.shape(feature)[1]
        num_neigh1 = tf.shape(feature)[2]
        d1 = feature.get_shape()[3].value
        f_concatcddelta = tf.concat([delta_xyz,delta_feature], axis=-1)  #[B, npoint, nsample, C+D]
        #
        line_activation = tf.layers.dense( f_concatcddelta, d1, activation=None, use_bias=False, name='line_%d'%(i))
        line_activation=tf.nn.leaky_relu(line_activation)  #[B, npoint, nsample, D]
        
        f_reshaped_line = tf.reshape(line_activation, shape=[-1, num_neigh1, d1])  #[B*npoint, nsample, D]
        att_maps = tf.nn.softmax(f_reshaped_line, axis=-1)
        att_maps=tf_util.dropout(att_maps, keep_prob=0.8, is_training=is_training, scope=scope)
        
        f_reshaped_feature = tf.reshape(feature, shape=[-1, num_neigh1, d1])  #[B*npoint, nsample, D]
        att_feature =f_reshaped_feature*att_maps    #[B*npoint, nsample,D]
        
        att_feature = tf.reshape(att_feature, [batch_size1, num_points1, num_neigh1, d1])
        
        feature=tf.add(feature,att_feature)
        
    return feature  

def att_pooling(feature_set, d_out, is_training, bn_decay, bn,scope,data_format,i):
    with tf.variable_scope(scope) as sc:
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name='fc_%d'%(i))
        att_activation=tf.nn.leaky_relu(att_activation)
        att_scores = tf.nn.softmax(att_activation, axis=-1)  #[B*npoint, nsample, d]axis=-1
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = tf_util.conv2d(f_agg, d_out, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='attpool_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format) 
        #f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
    return f_agg



'''
def CAM_Module(block_to_upsample, skip_connection, n_filters_keep,Channel,scope=None):
    
    """ 
    Channel relation module 
    """
    #Channel=192
    Channel=Channel+n_filters_keep
    l = slim.conv2d_transpose(block_to_upsample, n_filters_keep, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
    # Concatenate with skip connection
    inputs= tf.concat([l, skip_connection], axis=-1)
    
    GAP= tf.reduce_mean(inputs, [1, 2], keep_dims=True)
    
    net = slim.conv2d(GAP,Channel//8 , kernel_size=[1, 1])
    net = tf.nn.relu(net)
    net = slim.conv2d(net, Channel, kernel_size=[1, 1])
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    net = tf.add(inputs, net)
   
    return net
'''
 
def pointnet_nearest_interpolation(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor sparser than xyz1 
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        #_,idx = knn_point(1, xyz2,xyz1)    #(batch_size, npoint, nsample) nsample=k
        idx = tf.py_func(knn_query, [1, xyz2,xyz1], tf.int32)
        interpolated_points = group_point(points2, idx) # (batch_size, npoint, k, channel)
        interpolated_points = tf.squeeze(interpolated_points, axis=2)
        '''
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)
        '''
        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        '''
        if points1 is not None:
            inputs=tf.concat(axis=2, values=[interpolated_points, points1])   # B,ndataset1,nchannel1+nchannel2
            d2 = inputs.get_shape()[2].value
            GAP= tf.reduce_mean(inputs, [1], keep_dims=True)
            input_net =tf.layers.dense(GAP, max(32, d2//16), activation=None, use_bias=False, name='interpolated_%d'%(0))
            input_net = tf.nn.relu(input_net)
            input_net = tf.layers.dense(input_net, d2, activation=None, use_bias=False, name='interpolated_%d'%(1))
            input_net = tf.sigmoid(input_net)
            input_net= tf.multiply(inputs, input_net)
            new_points1 = tf.add(inputs,input_net)
            #new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        '''
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1

def nearest_interpolation(feature, sm_xyz,big_xyz):
    """
    :param feature: [B, N, d] input features matrix
    sm_xyz1: (batch_size, ndataset, c) float32 array, input points
    big_xyz2: (batch_size, npoint, c) float32 array, query points
    :param interp_idx: [B, up_num_points, 1] nearest neighbour index
    :return: [B, up_num_points, d] interpolated features matrix
    """
    feature = tf.squeeze(feature, axis=2)
    _,idx = knn_point(1, sm_xyz,big_xyz)    #(batch_size, npoint, nsample) nsample=k
    
    interpolated_features = group_point(feature, idx) # (batch_size, npoint, k, channel)
    
    #interpolated_features = tf.expand_dims(interpolated_features, axis=2)
    return interpolated_features
 
def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1

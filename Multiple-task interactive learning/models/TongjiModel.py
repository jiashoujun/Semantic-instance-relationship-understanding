import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from TongjiModel_util import pointnet_nearest_interpolation,pointresnet_module

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None):
    
    end_points = {}
    xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
 
    end_points['l0_xyz'] =xyz 
    feature=point_cloud   #(24, 4096, 6)
    
   # feature = tf.expand_dims(feature, axis=2) #(Batch,N ,1 C)  #4  (24, 4096, 1, 6)
    d_outshape = [64, 128, 256, 512,1024]  # feature dimension
    num_point = 40960
    num_points = [num_point//8, num_point//32, num_point//128, num_point//256, num_point//512]
    nsample1=16
 
    l1_xyz, l1_points, l1_indices = pointresnet_module(xyz, feature, npoint=num_points[0], nsample=nsample1, d_out=d_outshape[0],is_training=is_training, bn_decay=bn_decay, scope='layer1', bn=True, knn=True, use_nchw=False)
    
    l2_xyz, l2_points, l2_indices = pointresnet_module(l1_xyz, l1_points, npoint=num_points[1], nsample=nsample1, d_out=d_outshape[1],is_training=is_training, bn_decay=bn_decay, scope='layer2', bn=True, knn=True, use_nchw=False)
     
    l3_xyz, l3_points, l3_indices = pointresnet_module(l2_xyz, l2_points, npoint=num_points[2], nsample=nsample1, d_out=d_outshape[2],is_training=is_training, bn_decay=bn_decay, scope='layer3', bn=True, knn=True, use_nchw=False)
    
    l4_xyz, l4_points, l4_indices = pointresnet_module(l3_xyz, l3_points, npoint=num_points[3], nsample=nsample1,d_out=d_outshape[3],is_training=is_training, bn_decay=bn_decay, scope='layer4', bn=True, knn=True, use_nchw=False)
    
    l5_xyz, l5_points, l5_indices = pointresnet_module(l4_xyz, l4_points, npoint=num_points[4], nsample=nsample1,d_out=d_outshape[4],is_training=is_training, bn_decay=bn_decay, scope='layer5', bn=True, knn=True, use_nchw=False)
    

    # Feature Propagation layers
    l4_points = pointnet_nearest_interpolation(l4_xyz, l5_xyz, l4_points, l5_points, [768,768], is_training, bn_decay, scope='fa_layer0')
    l3_points = pointnet_nearest_interpolation(l3_xyz, l4_xyz, l3_points, l4_points, [512,512], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_nearest_interpolation(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
    l1_points = pointnet_nearest_interpolation(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer3')
    l0_points = pointnet_nearest_interpolation(xyz, l1_xyz, feature, l1_points, [128,128], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.9, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc3')

    return net, end_points

def get_weight_loss(logits, labels):
        
        num_per_class =np.array([20570173,13684381,5303097,2057941,72478,404608], dtype=np.int32)

        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        pre_cal_weights=np.expand_dims(ce_label_weight, axis=0)
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        
        logits = tf.reshape(logits, [-1, len(num_per_class)])
        labels = tf.reshape(labels, [-1])
        
        one_hot_labels = tf.one_hot(labels, depth=len(num_per_class))
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        classify_loss = tf.reduce_mean(weighted_losses)
        tf.summary.scalar('classify loss', classify_loss)
        tf.add_to_collection('losses', classify_loss)
        return classify_loss

def get_loss(pred, label):

    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)

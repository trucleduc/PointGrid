import tensorflow as tf
import tensorflow.contrib.slim as slim
import json
import numpy as np
from functools import partial
from scipy import stats
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, './utils'))
import tf_util

N = 16
K = 16
NUM_CATEGORY = 16
NUM_SEG_PART = 50+1
NUM_PER_POINT_FEATURES = 3
NUM_FEATURES = K * NUM_PER_POINT_FEATURES

batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, scope='bn', updates_collections=None)

def leak_relu(x, leak=0.1, scope=None):
    return tf.where(x >= 0, x, leak * x)



def pc2voxel(pc, pc_label):
    # Args:
    #     pc: size n x F where n is the number of points and F is feature size
    #     pc_label: size n x NUM_SEG_PART (one-hot encoding label)
    # Returns:
    #     voxel: N x N x N x K x (3+3)
    #     label: N x N x N x (K+1) x NUM_SEG_PART
    #     index: N x N x N x K
    #     class_weights: NUM_SEG_PART

    num_points = pc.shape[0]
    data = np.zeros((N, N, N, K, NUM_PER_POINT_FEATURES), dtype=np.float32)
    label = np.zeros((N, N, N, K+1, NUM_SEG_PART), dtype=np.float32)
    index = np.zeros((N, N, N, K), dtype=np.float32)
    class_counts = np.ones((NUM_SEG_PART), dtype=np.float32)
    xyz = pc[:, 0 : 3]
    centroid = np.mean(xyz, axis=0, keepdims=True)
    xyz -= centroid
    xyz /= np.amax(np.sqrt(np.sum(xyz ** 2, axis=1)), axis=0) * 1.05
    idx = np.floor((xyz + 1.0) / 2.0 * N)
    L = [[] for _ in range(N * N * N)]
    for p in range(num_points):
        k = int(idx[p, 0] * N * N + idx[p, 1] * N + idx[p, 2])
        L[k].append(p)
    for i in range(N):
      for j in range(N):
        for k in range(N):
          u = int(i * N * N + j * N + k)
          if not L[u]:
              data[i, j, k, :, :] = np.zeros((K, NUM_PER_POINT_FEATURES))
              label[i, j, k, :, :] = 0
              label[i, j, k, :, 0] = 1
              class_counts[0] += (K+1.0)
          elif (len(L[u]) >= K):
              choice = np.random.choice(L[u], size=K, replace=False)
              data[i, j, k, :, :] = pc[choice, :]
              label[i, j, k, 0 : K, :] = pc_label[choice, :]
              majority = np.argmax(np.sum(pc_label[L[u], :], axis=0))
              label[i, j, k, K, :] = 0
              label[i, j, k, K, majority] = 1
              index[i, j, k, :] = choice
              for s in range(K):
                  class_counts[np.argmax(pc_label[choice[s], :])] += 1.0
              class_counts[majority] += 1.0
          else:
              choice = np.random.choice(L[u], size=K, replace=True)
              data[i, j, k, :, :] = pc[choice, :]
              label[i, j, k, 0 : K, :] = pc_label[choice, :]
              majority = np.argmax(np.sum(pc_label[L[u], :], axis=0))
              label[i, j, k, K, :] = 0
              label[i, j, k, K, majority] = 1
              index[i, j, k, :] = choice
              for s in range(K):
                  class_counts[np.argmax(pc_label[choice[s], :])] += 1.0
              class_counts[majority] += 1.0
    data = np.reshape(data, (N, N, N, K * NUM_PER_POINT_FEATURES))
    class_weights = np.sum(class_counts) / class_counts
    class_weights = class_weights / np.sum(class_weights)
    return data, label, index, class_weights



def populateIntegerSegLabel(pc, voxel_label, index):
    # Args:
    #     pc: size n x F where n is the number of points and F is feature size
    #     voxel_label: size N x N x N x (K+1)
    #     index: size N x N x N x K
    # Returns:
    #     label: size n x 1

    num_points = pc.shape[0]
    label = np.zeros((num_points), dtype=np.int32)
    xyz = pc[:, 0 : 3]
    centroid = np.mean(xyz, axis=0, keepdims=True)
    xyz -= centroid
    xyz /= np.amax(np.sqrt(np.sum(xyz ** 2, axis=1)), axis=0) * 1.05
    idx = np.floor((xyz + 1.0) / 2.0 * N)
    L = [[] for _ in range(N * N * N)]
    for p in range(num_points):
        k = int(idx[p, 0] * N * N + idx[p, 1] * N + idx[p, 2])
        L[k].append(p)
    for i in range(N):
      for j in range(N):
        for k in range(N):
          u = int(i * N * N + j * N + k)
          if not L[u]:
              pass
          elif (len(L[u]) >= K):
              label[L[u]] = voxel_label[i, j, k, K]
              for s in range(K):
                  label[int(index[int(i), int(j), int(k), int(s)])] = int(voxel_label[int(i), int(j), int(k), int(s)])
          else:
              for s in range(K):
                  label[int(index[int(i), int(j), int(k), int(s)])] = int(voxel_label[int(i), int(j), int(k), int(s)])
    return label



def populateOneHotSegLabel(pc, voxel_label, index):
    # Args:
    #     pc: size n x F where n is the number of points and F is feature size
    #     voxel_label: size N x N x N x (K+1) x NUM_SEG_PART
    #     index: size N x N x N x K
    # Returns:
    #     label: size n x 1
    return populateIntegerSegLabel(pc, np.argmax(voxel_label, axis=4), index)



def get_model(pointgrid, is_training):
    # Args:
    #     pointgrid: of size B x N x N x N x NUM_FEATURES
    #     is_training: boolean tensor
    # Returns:
    #     pred_label: of size B x N x N x N x (K+1) x NUM_PART_SEG

    batch_size = pointgrid.get_shape()[0].value

    conv1 = tf_util.conv3d(pointgrid, 64, [5,5,5], scope='conv1', activation_fn=leak_relu, bn=True, is_training=is_training) # N
    conv2 = tf_util.conv3d(conv1, 64, [5,5,5], scope='conv2', activation_fn=leak_relu, stride=[2,2,2], bn=True, is_training=is_training) # N/2
    conv3 = tf_util.conv3d(conv2, 64, [5,5,5], scope='conv3', activation_fn=leak_relu, bn=True, is_training=is_training) # N/2
    conv4 = tf_util.conv3d(conv3, 128, [3,3,3], scope='conv4', activation_fn=leak_relu, stride=[2,2,2], bn=True, is_training=is_training) # N/4
    conv5 = tf_util.conv3d(conv4, 128, [3,3,3], scope='conv5', activation_fn=leak_relu, bn=True, is_training=is_training) # N/4
    conv6 = tf_util.conv3d(conv5, 256, [3,3,3], scope='conv6', activation_fn=leak_relu, stride=[2,2,2], bn=True, is_training=is_training) # N/8
    conv7 = tf_util.conv3d(conv6, 256, [3,3,3], scope='conv7', activation_fn=leak_relu, bn=True, is_training=is_training) # N/8
    conv8 = tf_util.conv3d(conv7, 512, [3,3,3], scope='conv8', activation_fn=leak_relu, stride=[2,2,2], bn=True, is_training=is_training) # N/16
    conv9 = tf_util.conv3d(conv8, 512, [1,1,1], scope='conv9', activation_fn=leak_relu, bn=True, is_training=is_training) # N/16

    deconv1 = tf_util.conv3d_transpose(conv9, 256, [3,3,3], scope='deconv1', activation_fn=leak_relu, bn=True, is_training=is_training, stride=[2,2,2], padding='SAME') # N/8
    conv7_deconv1 = tf.concat(axis=4, values=[conv7, deconv1])
    deconv2 = tf_util.conv3d(conv7_deconv1, 256, [3,3,3], scope='deconv2', activation_fn=leak_relu, bn=True, is_training=is_training) # N/8
    deconv3 = tf_util.conv3d_transpose(deconv2, 128, [3,3,3], scope='deconv3', activation_fn=leak_relu, bn=True, is_training=is_training, stride=[2,2,2], padding='SAME') # N/4
    conv5_deconv3 = tf.concat(axis=4, values=[conv5, deconv3])
    deconv4 = tf_util.conv3d(conv5_deconv3, 128, [3,3,3], scope='deconv4', activation_fn=leak_relu, bn=True, is_training=is_training) # N/4
    deconv5 = tf_util.conv3d_transpose(deconv4, 64, [3,3,3], scope='deconv5', activation_fn=leak_relu, bn=True, is_training=is_training, stride=[2,2,2], padding='SAME') # N/2
    conv3_deconv5 = tf.concat(axis=4, values=[conv3, deconv5])
    deconv6 = tf_util.conv3d(conv3_deconv5, 64, [5,5,5], scope='deconv6', activation_fn=leak_relu, bn=True, is_training=is_training) # N/2
    deconv7 = tf_util.conv3d_transpose(deconv6, 64, [5,5,5], scope='deconv7', activation_fn=leak_relu, bn=True, is_training=is_training, stride=[2,2,2], padding='SAME') # N
    conv1_deconv7 = tf.concat(axis=4, values=[conv1, deconv7])
    deconv8 = tf_util.conv3d(conv1_deconv7, 64, [5,5,5], scope='deconv8', activation_fn=leak_relu, bn=True, is_training=is_training) # N

    pred_label = tf_util.conv3d(deconv8, (K+1) * NUM_SEG_PART, [5,5,5], scope='pred_label', activation_fn=None, bn=False, is_training=is_training)
    pred_label = tf.reshape(pred_label, [batch_size, N, N, N, K+1, NUM_SEG_PART])

    return pred_label



def get_loss(pred_label, one_hot_label, class_weight):
    pred_label = tf.reshape(pred_label, [-1, NUM_SEG_PART])
    one_hot_label = tf.reshape(one_hot_label, [-1, NUM_SEG_PART])
    #class_weight = tf.reduce_sum(class_weight, axis=0)
    #weights = tf.reduce_sum(class_weight * one_hot_label, axis=1)
    per_instance_seg_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_label, labels=one_hot_label)
    seg_loss = tf.constant(1000.0, dtype=tf.float32) * tf.reduce_mean(per_instance_seg_loss)
    return seg_loss

def intersection_over_union(pred_seg, integer_seg_label):
    iou, counts = 0.0, 0.0
    for i in range(1, NUM_SEG_PART):
        intersection = np.sum(np.logical_and(pred_seg == i, integer_seg_label == i))
        union = np.sum(np.logical_or(pred_seg == i, integer_seg_label == i))
        if (union > 0):
            counts += 1.0
            iou += (float(intersection) / float(union))
    iou /= counts
    return iou

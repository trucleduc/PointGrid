import argparse
import tensorflow as tf
import json
import numpy as np
import scipy.io
import os
import sys
import glob
from skimage import measure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import network as model

parser = argparse.ArgumentParser()
FLAGS = parser.parse_args()


# DEFAULT SETTINGS
gpu_to_use = 0
output_dir = os.path.join(BASE_DIR, './test_results')

# MAIN SCRIPT
batch_size = 1               # DO NOT CHANGE

def get_file_name(file_path):
    parts = file_path.split('/')
    part = parts[-1]
    parts = part.split('.')
    return parts[0]

TESTING_FILE_LIST = [get_file_name(file_name) for file_name in glob.glob('../data/ShapeNet/test/' + '*.mat')]

category2name = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table']

color_map = [[1.000000, 1.000000, 1.000000],
             [0.000000, 0.000000, 1.000000],
             [1.000000, 0.000000, 0.000000],
             [0.000000, 1.000000, 0.000000],
             [1.000000, 0.103448, 0.724138],
             [1.000000, 0.827586, 0.000000],
             [0.000000, 0.517241, 0.965517],
             [0.000000, 0.551724, 0.275862],
             [0.655172, 0.379310, 0.241379],
             [0.310345, 0.000000, 0.413793],
             [0.000000, 1.000000, 0.965517],
             [0.241379, 0.482759, 0.551724],
             [0.931034, 0.655172, 1.000000],
             [0.827586, 1.000000, 0.586207],
             [0.724138, 0.310345, 1.000000],
             [0.896552, 0.103448, 0.344828],
             [0.517241, 0.517241, 0.000000],
             [0.000000, 1.000000, 0.586207],
             [0.379310, 0.000000, 0.172414],
             [0.965517, 0.517241, 0.068966],
             [0.793103, 1.000000, 0.000000],
             [0.172414, 0.241379, 0.000000],
             [0.000000, 0.206897, 0.758621],
             [1.000000, 0.793103, 0.517241],
             [0.000000, 0.172414, 0.379310],
             [0.620690, 0.448276, 0.551724],
             [0.310345, 0.724138, 0.068966],
             [0.620690, 0.758621, 1.000000],
             [0.586207, 0.620690, 0.482759],
             [1.000000, 0.482759, 0.689655],
             [0.620690, 0.034483, 0.000000],
             [1.000000, 0.724138, 0.724138],
             [0.517241, 0.379310, 0.793103],
             [0.620690, 0.000000, 0.448276],
             [0.517241, 0.862069, 0.655172],
             [1.000000, 0.000000, 0.965517],
             [0.000000, 0.827586, 1.000000],
             [1.000000, 0.448276, 0.344828],
             [0.344828, 0.241379, 0.206897],
             [0.000000, 0.241379, 0.206897],
             [0.862069, 0.379310, 0.862069],
             [0.379310, 0.448276, 0.689655],
             [0.724138, 0.793103, 0.172414],
             [0.068966, 0.689655, 0.655172],
             [0.379310, 0.068966, 0.000000],
             [0.172414, 0.000000, 0.172414],
             [0.344828, 0.000000, 0.793103],
             [0.586207, 0.758621, 0.793103],
             [0.827586, 0.620690, 0.137931],
             [0.517241, 0.689655, 0.344828],
             [0.896552, 0.931034, 0.724138]]

lines = [line.rstrip('\n') for line in open('sphere.txt')]
nSphereVertices = int(lines[0])
sphereVertices = np.zeros((nSphereVertices, 3))
for i in range(nSphereVertices):
    coordinates = lines[i + 1].split()
    for j in range(len(coordinates)):
        sphereVertices[i, j] = float(coordinates[j])
nSphereFaces = int(lines[nSphereVertices + 1])
sphereFaces = np.zeros((nSphereFaces, 3))
for i in range(nSphereFaces):
    indices = lines[i + nSphereVertices + 2].split()
    for j in range(len(coordinates)):
        sphereFaces[i, j] = int(indices[j])

def output_color_point_cloud(data, seg, out_file, r=0.01):
    count = 0
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            for j in range(nSphereVertices):
                f.write('v %f %f %f %f %f %f\n' % \
                        (data[i][0] + sphereVertices[j][0] * r, data[i][1] + sphereVertices[j][1] * r, data[i][2] + sphereVertices[j][2] * r, color[0], color[1], color[2]))
            for j in range(nSphereFaces):
                f.write('f %d %d %d\n' % (count + sphereFaces[j][0], count + sphereFaces[j][1], count + sphereFaces[j][2]))
            count += nSphereVertices

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def placeholder_inputs():
    pointgrid_ph = tf.placeholder(tf.float32, shape=(batch_size, model.N, model.N, model.N, model.NUM_FEATURES))
    cat_label_ph = tf.placeholder(tf.float32, shape=(batch_size, model.NUM_CATEGORY))
    seg_label_ph = tf.placeholder(tf.float32, shape=(batch_size, model.N, model.N, model.N, model.K+1, model.NUM_SEG_PART))
    return pointgrid_ph, cat_label_ph, seg_label_ph

def integer_label_to_one_hot_label(integer_label):
    one_hot_label = np.zeros((integer_label.shape[0], model.NUM_SEG_PART))
    for i in range(integer_label.shape[0]):
        one_hot_label[i, integer_label[i]] = 1
    return one_hot_label

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_checkpoint(checkpoint_dir, session, var_list=None):
    print(' [*] Loading checkpoint...')
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    try:
        restorer = tf.train.Saver(var_list)
        restorer.restore(session, ckpt_path)
        print(' [*] Loading successful! Copy variables from % s' % ckpt_path)
        return True
    except:
        print(' [*] No suitable checkpoint!')
        return False

def predict():
    is_training = False
    
    with tf.device('/gpu:'+str(gpu_to_use)):
        pointgrid_ph, cat_label_ph, seg_label_ph = placeholder_inputs()
        is_training_ph = tf.placeholder(tf.bool, shape=())

        # model
        pred_cat, pred_seg = model.get_model(pointgrid_ph, is_training=is_training_ph)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        flog = open(os.path.join(output_dir, 'log.txt'), 'w')

        # Restore variables from disk.
        ckpt_dir = './train_results/trained_models'
        if not load_checkpoint(ckpt_dir, sess):
            exit()

        if not os.path.exists('../data/ShapeNet/test-PointGrid'):
            os.mkdir('../data/ShapeNet/test-PointGrid')
        
        avg_cat_accuracy = 0.0
        cat_accuracy = np.zeros((model.NUM_CATEGORY), dtype=np.float32)
        cat_obj = np.zeros((model.NUM_CATEGORY), dtype=np.int32)
        avg_iou = 0.0
        cat_iou = np.zeros((model.NUM_CATEGORY), dtype=np.float32)
        for loop in range(len(TESTING_FILE_LIST)):
            mat_content = scipy.io.loadmat('../data/ShapeNet/test/' + TESTING_FILE_LIST[loop] + '.mat')
            pc = mat_content['points']
            labels = mat_content['labels']
            category = mat_content['category'][0][0]
            cat_label = model.integer_label_to_one_hot_label(category)
            category = int(category)
            cat_obj[category] += 1
            seg_label = integer_label_to_one_hot_label(labels)
            pointgrid, pointgrid_label, index, _ = model.pc2voxel(pc, seg_label)
            pointgrid = np.expand_dims(pointgrid, axis=0)
            cat_label = np.expand_dims(cat_label, axis=0)
            pointgrid_label = np.expand_dims(pointgrid_label, axis=0)
            feed_dict = {
                         pointgrid_ph: pointgrid,
                         cat_label_ph: cat_label,
                         seg_label_ph: pointgrid_label,
                         is_training_ph: is_training,
                        }
            pred_cat_val, pred_seg_val = sess.run([pred_cat, pred_seg], feed_dict = feed_dict)
            pred_cat_val = np.argmax(pred_cat_val[0, :], axis=0)
            pred_seg_val = pred_seg_val[0, :, :, :, :, :]
            avg_cat_accuracy += (pred_cat_val == category)
            cat_accuracy[category] += (pred_cat_val == category)
            pred_point_label = model.populateOneHotSegLabel(pc, pred_seg_val, index)
            purify = True
            knn = 5
            if purify == True:
                pre_label = pred_point_label
                for i in range(pc.shape[0]):
                    idx = np.argsort(np.sum((pc[i, :] - pc) ** 2, axis=1))
                    j, L = 0, []
                    for _ in range(knn):
                        if (idx[j] == i):
                            j += 1
                        L.append(pre_label[idx[j]])
                        j += 1
                    majority = max(set(L), key=L.count)
                    if (pre_label[i] == 0 or len(set(L)) == 1):
                        pred_point_label[i] = majority
            iou = model.intersection_over_union(pred_point_label, labels)
            avg_iou += iou
            cat_iou[category] += iou
            if (cat_obj[category] <= 3):
                output_color_point_cloud(pc, pred_point_label, '../data/ShapeNet/test-PointGrid/' + category2name[category] + '_' + str(cat_obj[category]) + '.obj')
            printout(flog, '%d/%d %s' % ((loop+1), len(TESTING_FILE_LIST), TESTING_FILE_LIST[loop]))
            printout(flog, '----------')

        avg_cat_accuracy /= float(np.sum(cat_obj))
        avg_iou /= float(np.sum(cat_obj))
        printout(flog, 'Average classification accuracy: %f' % avg_cat_accuracy)
        printout(flog, 'Average IoU: %f' % avg_iou)
        printout(flog, 'CategoryName, CategorySize, ClassificationAccuracy, SegmentationIoU')
        for i in range(model.NUM_CATEGORY):
            cat_accuracy[i] /= float(cat_obj[i])
            cat_iou[i] /= float(cat_obj[i])
            printout(flog, '\t%s (%d): %f, %f' % (category2name[i], cat_obj[i], cat_accuracy[i], cat_iou[i]))



with tf.Graph().as_default():
    predict()

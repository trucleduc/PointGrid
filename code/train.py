import argparse
import subprocess
import tensorflow as tf
import threading
import numpy as np
import scipy.io
from datetime import datetime
import json
import os
import sys
import glob
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import network as model

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--epoch', type=int, default=200, help='Epoch to run [default: 200]')
parser.add_argument('--output_dir', type=str, default='train_results', help='Directory that stores all training logs and trained models')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
FLAGS = parser.parse_args()

# MAIN SCRIPT
batch_size = FLAGS.batch
output_dir = FLAGS.output_dir

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

print('#### Batch Size: {0}'.format(batch_size))
print('#### Training using GPU: {0}'.format(FLAGS.gpu))

LEARNING_RATE = 1e-4
TRAINING_EPOCHES = FLAGS.epoch
print('### Training epoch: {0}'.format(TRAINING_EPOCHES))

def get_file_name(file_path):
    parts = file_path.split('/')
    part = parts[-1]
    parts = part.split('.')
    return parts[0]

TRAINING_FILE_LIST = [get_file_name(file_name) for file_name in glob.glob('../data/ShapeNet/train/' + '*.mat')]

MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(output_dir, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER =  os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def integer_label_to_one_hot_label(integer_label):
    one_hot_label = np.zeros((integer_label.shape[0], model.NUM_SEG_PART))
    for i in range(integer_label.shape[0]):
        one_hot_label[i, integer_label[i]] = 1
    return one_hot_label

def load_and_enqueue(sess, enqueue_op, pointgrid_ph, label_ph, class_weight_ph):
    for epoch in range(1000 * TRAINING_EPOCHES):
        train_file_idx = np.arange(0, len(TRAINING_FILE_LIST))
        np.random.shuffle(train_file_idx)
        for loop in range(len(TRAINING_FILE_LIST)):
            mat_content = scipy.io.loadmat('../data/ShapeNet/train/' + TRAINING_FILE_LIST[train_file_idx[loop]] + '.mat')
            pc = mat_content['points']
            labels = mat_content['labels']
            category = mat_content['category'][0][0]
            one_hot_label = integer_label_to_one_hot_label(labels)
            pc = model.rotate_pc(pc)
            pointgrid, label, _, class_weights = model.pc2voxel(pc, one_hot_label)
            sess.run(enqueue_op, feed_dict={pointgrid_ph: pointgrid, label_ph: label, class_weight_ph: class_weights})

def placeholder_inputs():
    pointgrid_ph = tf.placeholder(tf.float32, shape=(model.N, model.N, model.N, model.NUM_FEATURES))
    label_ph = tf.placeholder(tf.float32, shape=(model.N, model.N, model.N, model.K+1, model.NUM_SEG_PART))
    class_weight_ph = tf.placeholder(tf.float32, shape=(model.NUM_SEG_PART))
    return pointgrid_ph, label_ph, class_weight_ph

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

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, target=None, args=None):
        super(StoppableThread, self).__init__(target=target, args=args)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointgrid_ph, label_ph, class_weight_ph = placeholder_inputs()
            is_training_ph = tf.placeholder(tf.bool, shape=())

            queue = tf.FIFOQueue(capacity=20*batch_size, dtypes=[tf.float32, tf.float32, tf.float32],\
                                                         shapes=[[model.N, model.N, model.N, model.NUM_FEATURES],\
                                                                 [model.N, model.N, model.N, model.K+1, model.NUM_SEG_PART],\
                                                                 [model.NUM_SEG_PART]])
            enqueue_op = queue.enqueue([pointgrid_ph, label_ph, class_weight_ph])
            dequeue_pointgrid, dequeue_label, dequeue_class_weight = queue.dequeue_many(batch_size)

            # model
            pred_label = model.get_model(dequeue_pointgrid, is_training=is_training_ph)

            # loss
            loss = model.get_loss(pred_label, dequeue_label, dequeue_class_weight)

            # optimization
            total_var = tf.trainable_variables()
            step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, var_list=total_var)

        # write logs to the disk
        flog = open(os.path.join(LOG_STORAGE_PATH, 'log.txt'), 'w')

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        ckpt_dir = './train_results/trained_models'
        if not load_checkpoint(ckpt_dir, sess):
            sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')

        fcmd = open(os.path.join(LOG_STORAGE_PATH, 'cmd.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()

        def train_one_epoch(epoch_num):
            is_training = True

            num_data = len(TRAINING_FILE_LIST)
            num_batch = num_data // batch_size
            loss_acc = 0.0
            display_mark = num_batch // 4
            for i in range(num_batch):
                _, loss_val = sess.run([step, loss], feed_dict={is_training_ph: is_training})
                loss_acc += loss_val

                if ((i+1) % display_mark == 0):
                    printout(flog, 'Epoch %d/%d - Iter %d/%d' % (epoch_num+1, TRAINING_EPOCHES, i+1, num_batch))
                    printout(flog, 'Segmentation Loss: %f' % (loss_acc / (i+1)))

            printout(flog, '\tMean Segmentation Loss: %f' % (loss_acc / num_batch))

        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)

        coord = tf.train.Coordinator()
        for num_thread in range(16):
            t = StoppableThread(target=load_and_enqueue, args=(sess, enqueue_op, pointgrid_ph, label_ph, class_weight_ph))
            t.setDaemon(True)
            t.start()
            coord.register_thread(t)

        for epoch in range(TRAINING_EPOCHES):
            printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch+1, TRAINING_EPOCHES))

            train_one_epoch(epoch)

            if (epoch+1) % 1 == 0:
                cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch+1)+'.ckpt'))
                printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)

            flog.flush()
        flog.close()

if __name__=='__main__':
    train()

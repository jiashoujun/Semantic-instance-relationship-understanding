import json
import argparse
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import time
from tqdm import tqdm
from tongji_dataset_grid import TONGJI
from sklearn.metrics import confusion_matrix
from metrics import IoU_from_confusions

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seg_label_to_cat = {0: 'steelframe', 1: 'keykeel', 2: 'aluminumbar', 3: 'column', 4: 'aluminumplate', 5: 'glass'}

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data', type=str, default='../data/Scannet', help='Root for dataset')
parser.add_argument('--model', default='TongjiModel', help='Model name')
parser.add_argument('--log_dir', default=None, help='Log dir [default: None]')
parser.add_argument('--pretrain_dir', default=None, help='Pretrain dir [default: None]')
parser.add_argument('--num_point', type=int, default=4096*10, help='Point number [default: 8192]')
parser.add_argument('--num_buffer', type=int, default=1024,
                    help='Buffer point number, work only if in_radius is 0 [default: 1024]')
parser.add_argument('--in_radius', type=float, default=0,
                    help='Radius of chopped area, work only if it larger than 0 [default: 0]')
parser.add_argument('--epoch_sample', type=int, default=3000, help='Number of steps per epochs [default: 4800]')
parser.add_argument('--validation_size', type=int, default=150,
                    help='Number of validation examples per epoch [default: 100]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 500]')
parser.add_argument('--from_epoch', type=int, default=0,
                    help='Epoch to run from (for restoring from checkpoints) [default: 0]')
parser.add_argument('--snapshot_gap', type=int, default=20, help='Gap for voting test [default: 20]')
parser.add_argument('--batch_size', type=int, default=5, help='Batch size during training [default: 4]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=400000, help='Decay step for lr decay [default: 400000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')

parser.add_argument('--input_threads', type=int, default=8,
                    help='Number of CPU threads for the input pipeline [default: 8]')
parser.add_argument('--first_subsampling_dl', type=float, default=0.060,
                    help='Voxel size for grid sampling [default: 0.04]')
parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
parser.add_argument('--trainval', action='store_true', help='Train with both train and valid sets [default: False]')
parser.add_argument('--debug', action='store_true')

FLAGS = parser.parse_args()

FLAGS.epoch_steps = FLAGS.epoch_sample // FLAGS.batch_size

if FLAGS.debug:
    FLAGS.epoch_steps = 50
    FLAGS.snapshot_gap = 1
    FLAGS.batch_size = 2

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
FROM_EPOCH = FLAGS.from_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
test_area = FLAGS.test_area
if not os.path.exists('shengyilog/'): os.mkdir('shengyilog/')
if FLAGS.log_dir is None:
    LOG_DIR = 'shengyilog/test/'
else:
    LOG_DIR = 'shengyilog/' + FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = '../models/' + FLAGS.model + '.py'
PointModel = '../utils/' + 'TongjiModel_util.py'
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp %s %s' % (PointModel, LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
    json.dump(FLAGS.__dict__, f, indent=2)

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# HOSTNAME = socket.gethostname()

NUM_CLASSES = 6
validation_probs = None
val_proportions = None


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def count_trainable_vars():
    total_parameters = 0
    for variable in tf.trainable_variables():
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters
 
    print("Total number of trainable parameters-----------------------------------------: %d" % total_parameters)

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):

            dataset =TONGJI('yixueyuan_7') 
            dataset.load_sub_sampled_clouds(FLAGS.first_subsampling_dl)
           
            gen_function_train, gen_types, gen_shapes = dataset.get_batch_gen('training')
            gen_function_val, _, _ = dataset.get_batch_gen('validation')

            train_data = tf.data.Dataset.from_generator(gen_function_train, gen_types, gen_shapes)
            val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)

            train_data = train_data.batch(FLAGS.batch_size)
            val_data = val_data.batch(FLAGS.batch_size)
            map_func = dataset.get_tf_mapping2()

            train_data = train_data.map(map_func=map_func)
            val_data = val_data.map(map_func=map_func)

            train_data = train_data.prefetch(FLAGS.batch_size)
            val_data = val_data.prefetch(FLAGS.batch_size)

            # create a iterator of the correct shape and type
            iter = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            flat_inputs = iter.get_next()

            # create the initialisation operations
            train_init_op = iter.make_initializer(train_data)
            val_init_op = iter.make_initializer(val_data)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(FROM_EPOCH * FLAGS.epoch_steps)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss ---")
            # Get model and loss
            points = flat_inputs[0]
            print(points.shape)

            point_labels = flat_inputs[1]

            pred, end_points = MODEL.get_model(points, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            #loss = MODEL.get_loss(pred, point_labels)
            loss = MODEL.get_weight_loss(pred, point_labels)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(point_labels))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE * NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=50)

        # Create a session
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        config_proto.allow_soft_placement = True
        config_proto.log_device_placement = False
        sess = tf.Session(config=config_proto)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

        if FLAGS.pretrain_dir is not None:
            saver.restore(sess, FLAGS.pretrain_dir)
            print('Loading model from %s ...' % FLAGS.pretrain_dir)
        else:
            init = tf.global_variables_initializer()  # Init variables
            sess.run(init, {is_training_pl: True})
            print('Training from scratch ...')
        count_trainable_vars()
        ops = {'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'labels': point_labels,
               'point_inds': flat_inputs[-2],
               'cloud_inds': flat_inputs[-1],
               'end_points': end_points}

        best_iou = 0
        best_iou_whole = 0

        for epoch in range(FROM_EPOCH, MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            start_time = time.time()
            train_one_epoch(sess, ops, train_writer, train_init_op)
            end_time = time.time()
            log_string('one epoch time: %.4f' % (end_time - start_time))
            remain_time = (end_time - start_time) * (MAX_EPOCH - 1 - epoch)
            m, s = divmod(remain_time, 60)
            h, m = divmod(m, 60)
            if s != 0:
                log_string("Remaining training time = %d hours %d minutes %d seconds\n" % (h, m, s))
            iou, iou_whole = eval_one_epoch(pred, ops, sess, val_init_op, dataset, epoch % FLAGS.snapshot_gap == 0)

            if iou > best_iou:
                best_iou = iou

            if iou_whole > best_iou_whole:
                best_iou_whole = iou_whole
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'best_epoch_' + str(epoch) + '_model.ckpt'))
                log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            save_path = saver.save(sess, os.path.join(LOG_DIR, "latest_" + str(epoch) + "_model.ckpt"))
            log_string("Model saved in file: %s\n" % save_path)
            log_string('Best chopped class avg mIOU is: %.3f' % best_iou)
            if best_iou_whole > 0:
                log_string('Best voting whole scene class avg mIOU is: %.3f  \n' % best_iou_whole)


def train_one_epoch(sess, ops, train_writer, train_init_op):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    log_string('---- TRAINING ----')
    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_iou_deno = 0
    sess.run(train_init_op)
    num_steps = 0
    with tqdm(total=FLAGS.epoch_steps) as pbar:
        while True:
            try:
                feed_dict = {ops['is_training_pl']: is_training, }
                summary, step, _, loss_val, pred_val, batch_label = \
                    sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],
                              ops['pred'], ops['labels']], feed_dict=feed_dict)

                train_writer.add_summary(summary, step)
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum(pred_val == batch_label)
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                iou_deno = 0
                for l in range(NUM_CLASSES):
                    iou_deno += np.sum((pred_val == l) | (batch_label == l))
                total_iou_deno += iou_deno
                loss_sum += loss_val
                num_steps += 1
                pbar.update(1)
            except tf.errors.OutOfRangeError:
                sess.run(train_init_op)
                break

    log_string('Training Loss: %f' % (loss_sum / num_steps))
    log_string('Training Accuracy: %f' % (total_correct / float(total_seen) * 100))
    log_string('Training IoU: %f' % (total_correct / float(total_iou_deno) * 100))


def eval_one_epoch(pred, inputs, sess, val_init_op, dataset, vote=False):
    """ ops: dict mapping from string to tf ops """
    log_string('---- EVALUATION ----')
    # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
    val_smooth = 0.95

    prob_logits = tf.nn.softmax(pred[:, :, 0:])

    
    sess.run(val_init_op)
    gt_classes = [0 for _ in range(NUM_CLASSES)]
    positive_classes = [0 for _ in range(NUM_CLASSES)]
    true_positive_classes = [0 for _ in range(NUM_CLASSES)]
    val_total_correct = 0
    val_total_seen = 0

    for _ in tqdm(range(FLAGS.validation_size), total=FLAGS.validation_size):
        try:
            ops = (prob_logits,
                   inputs['labels'],
                   inputs['point_inds'],
                   inputs['cloud_inds'])

            stacked_probs, labels, point_inds, cloud_inds = sess.run(ops, {inputs['is_training_pl']: False})
            stacked_probs = stacked_probs.reshape(-1, NUM_CLASSES)
            labels = labels.reshape(-1)

            pred2 = np.argmax(stacked_probs, 1)

            pred_valid = pred2
            labels_valid = labels

            correct = np.sum(pred_valid == labels_valid)
            val_total_correct += correct
            val_total_seen += len(labels_valid)

            conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, NUM_CLASSES, 1))
            gt_classes += np.sum(conf_matrix, axis=1)
            positive_classes += np.sum(conf_matrix, axis=0)
            true_positive_classes += np.diagonal(conf_matrix)


        except tf.errors.OutOfRangeError:
            break

    iou_list = []
    for n in range(0, NUM_CLASSES, 1):
        iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
        iou_list.append(iou)
    mIoU = sum(iou_list) / float(NUM_CLASSES)

    log_string('Eval point avg class IoU: %.3f' % (mIoU))
    log_string('eval accuracy: %.3f' % (val_total_correct / float(val_total_seen)))

    # Voting test
    mIoU_vote = 0
    
    return mIoU, mIoU_vote


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()

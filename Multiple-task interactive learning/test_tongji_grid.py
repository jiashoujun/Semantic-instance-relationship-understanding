import os
import sys
from os import makedirs
from os.path import exists, join
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from ply_helper import read_ply, write_ply
from sklearn.metrics import confusion_matrix
from metrics import IoU_from_confusions
import json
import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import time
from pathlib import Path
from helper_tool import ConfigTONGJI as cfg
from helper_tool import DataProcessing as DP
from tongji_dataset_grid import TONGJI

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data', type=str, default='../data/Scannet', help='Root for dataset')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 4]')
parser.add_argument('--model', default='TongjiModel', help='Model name')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--num_votes', type=int, default=100, help='Aggregate scores from multiple test [default: 100]')
parser.add_argument('--split', type=str, default='validation', help='[validation/test]')
parser.add_argument('--first_subsampling_dl', type=float, default=0.060, help='Voxel size for grid sampling [default: 0.04]')
parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
parser.add_argument('--saving', action='store_true', help='Whether save test results')
parser.add_argument('--debug', action='store_true', help='Whether save test results')
FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

validation_size =80
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = cfg.num_points
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
test_area = FLAGS.test_area
MODEL = importlib.import_module(FLAGS.model)  # import network module
NUM_CLASSES = 6

def log_out(out_str, log_f_out):
    log_f_out.write(out_str + '\n')
    log_f_out.flush()
    print(out_str)

class ModelTester:
    def __init__(self, pred, num_classes, saver, restore_snap=None):

        self.saver = saver
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        cProto.allow_soft_placement = True
        cProto.log_device_placement = False
        self.sess = tf.Session(config=cProto)
        self.Log_file = open('log_test_' + str('validation') + '.txt', 'a')
        if (restore_snap is not None):
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)
        else:
            self.sess.run(tf.global_variables_initializer())

        # Add a softmax operation for predictions
        self.prob_logits = tf.nn.softmax(pred[:, :, 0:])
        self.num_classes = num_classes
        
   

    def test_cloud_segmentation_on_val(self, input, dataset, val_init_op, num_votes=100, saving=True):

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(val_init_op)

        # Initiate global prediction over test clouds
        nc_model = self.num_classes 
        self.test_probs = [np.zeros((l.shape[0], nc_model), dtype=np.float32)
                           for l in dataset.input_labels['validation']]

        # Number of points per class in validation set
        val_proportions = np.zeros(nc_model, dtype=np.float32)
        i = 0
        for label_value in dataset.label_values:
            if label_value not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_value)
                                             for labels in dataset.label_values])
                i += 1

        # Test saving path
        if saving:
            saving_path ='shoujun' 
            test_path = join('test', saving_path)
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'val_predictions')):
                makedirs(join(test_path, 'val_predictions'))
            if not exists(join(test_path, 'val_probs')):
                makedirs(join(test_path, 'val_probs'))
        else:
            test_path = None

        step_id = 0
        epoch_id = 0
        last_min = -0.5
        mean_dt = np.zeros(2)
        start_time = time.time()
        while last_min < num_votes:
            try:
                # Run one step of the model.
                
                ops = (self.prob_logits,
                       input['labels'],
                       input['point_inds'],
                       input['cloud_inds'])
                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {input['is_training_pl']: False})
                
                stacked_probs = stacked_probs.reshape(-1, NUM_CLASSES)
                stacked_labels = stacked_labels.reshape(-1)

                correct = np.sum(np.argmax(stacked_probs, axis=1) == stacked_labels)
                acc = correct / float(np.prod(np.shape(stacked_labels)))
                print('step' + str(step_id) + ' acc:' + str(acc))
                stacked_probs = np.reshape(stacked_probs, [BATCH_SIZE, cfg.num_points,
                                                           cfg.num_classes])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    p_idx = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][p_idx] = test_smooth * self.test_probs[c_i][p_idx] + (1 - test_smooth) * probs
                step_id += 1
                

            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_possibility['validation'])
                log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.Log_file)

                if last_min + 1 < new_min:

                    # Update last_min
                    last_min += 1

                    # Show vote results (On subcloud so it is not the good values here)
                    log_out('\nConfusion on sub clouds', self.Log_file)
                    confusion_list = []

                    num_val = len(dataset.input_labels['validation'])

                    for i_test in range(num_val):
                        probs = self.test_probs[i_test]
                        preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                        labels = dataset.input_labels['validation'][i_test]

                        # Confs
                        confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = DP.IoU_from_confusions(C)
                    m_IoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * m_IoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    log_out(s + '\n', self.Log_file)

                    if int(np.ceil(new_min)) % 1 == 0:

                        # Project predictions
                        log_out('\nReproject Vote #{:d}'.format(int(np.floor(new_min))), self.Log_file)
                        proj_probs_list = []

                        for i_val in range(num_val):
                            # Reproject probs back to the evaluations points
                            proj_idx = dataset.val_proj[i_val]
                            probs = self.test_probs[i_val][proj_idx, :]
                            proj_probs_list += [probs]

                        # Show vote results
                        log_out('Confusion on full clouds', self.Log_file)
                        confusion_list = []
                        acc_list=[]
                        for i_test in range(num_val):
                            # Get the predicted labels
                            preds = dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)

                            # Confusion
                            labels = dataset.val_labels[i_test]
                            acc = np.sum(preds == labels) / len(labels)
                            acc_list.append(acc)
                            log_out(dataset.input_names['validation'][i_test] + ' Acc:' + str(acc), self.Log_file)

                            confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]
                            name = dataset.input_names['validation'][i_test] + '.txt'
                            print(name)
                            np.savetxt(join(test_path, 'val_predictions', name), preds, fmt='%d')
                            #name = dataset.input_names['validation'][i_test] + '.ply'
                            #write_ply(join(test_path, 'val_predictions', name), [preds, labels], ['pred', 'label'])

                        log_out(' mean Acc:' + str(np.mean(acc_list)), self.Log_file)
                        # Regroup confusions
                        C = np.sum(np.stack(confusion_list), axis=0)

                        IoUs = DP.IoU_from_confusions(C)
                        m_IoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * m_IoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        log_out('-' * len(s), self.Log_file)
                        log_out(s, self.Log_file)
                        log_out('-' * len(s) + '\n', self.Log_file)
                        print('finished \n')
                        self.sess.close()
                        return

                self.sess.run(val_init_op)#self.sess.run(dataset.val_init_op)
                epoch_id += 1
                step_id = 0
                continue
               
        return

def val():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            dataset = TONGJI('shengming_2')
            dataset.load_sub_sampled_clouds(FLAGS.first_subsampling_dl)
            #cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
       
            
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
            val_data  = val_data.prefetch(FLAGS.batch_size)
            
              # create a iterator of the correct shape and type
            iter = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            flat_inputs = iter.get_next()

            # create the initialisation operations
            train_init_op = iter.make_initializer(train_data)
            val_init_op = iter.make_initializer(val_data)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
         
            points = flat_inputs[0]
            

            point_labels = flat_inputs[1]
           
            pred, end_points = MODEL.get_model(points, is_training_pl, NUM_CLASSES)
            saver = tf.train.Saver()

            input = {
                    'is_training_pl': is_training_pl,
                    'pred': pred,
                    'labels': point_labels,
                    'point_inds': flat_inputs[-2],
                    'cloud_inds': flat_inputs[-1]}

            tester = ModelTester(pred, NUM_CLASSES, saver, MODEL_PATH )

            tester.test_cloud_segmentation_on_val(input, dataset, val_init_op)
          

if __name__ == "__main__":
    val()

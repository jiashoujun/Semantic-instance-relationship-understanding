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
parser.add_argument('--gpu', type=int, default=3, help='GPU to use [default: GPU 0]')
parser.add_argument('--data', type=str, default='../data/Scannet', help='Root for dataset')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 4]')
parser.add_argument('--num_points', type=int, default=4096*11, help='Batch Size during training [default: 4]')
parser.add_argument('--model', default='TongjiModel', help='Model name')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--num_votes', type=int, default=100, help='Aggregate scores from multiple test [default: 100]')
parser.add_argument('--split', type=str, default='validation', help='[validation/test]')
parser.add_argument('--first_subsampling_dl', type=float, default=0.060, help='Voxel size for grid sampling [default: 0.06]')
parser.add_argument('--saving', action='store_true', help='Whether save test results')
parser.add_argument('--debug', action='store_true', help='Whether save test results')
FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)



BATCH_SIZE = FLAGS.batch_size
NUM_POINT = cfg.num_points
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model)  
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
        
   

    def test_cloud_segmentation_on_val(self, input, dataset, test_init_op, num_votes=50):

       # Smoothing parameter for votes
        test_smooth = 0.98

        # Initialise iterator with train data
        self.sess.run(test_init_op)

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'predictions')) if not exists(join(test_path, 'predictions')) else None
        makedirs(join(test_path, 'probs')) if not exists(join(test_path, 'probs')) else None

        self.test_probs = [np.zeros((l.data.shape[0], self.num_classes), dtype=np.float16)
                           for l in dataset.input_trees['test']]

        #####################
        # Network predictions
        #####################

        step_id = 0
        epoch_id = 0
        last_min = -0.5
      
        while last_min < num_votes:
            try: 
                ops = (self.prob_logits,
                        input['labels'],
                        input['point_inds'],
                        input['cloud_inds'],)
                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {input['is_training_pl']: False})
                stacked_probs = np.reshape(stacked_probs, [cfg.val_batch_size,cfg.num_points,
                                                           cfg.num_classes])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    inds = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                step_id += 1
                log_out('Epoch {:3d}, step {:3d}. min possibility = {:.1f}'.format(epoch_id, step_id, np.min(
                    dataset.min_possibility['test'])), self.Log_file)
            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_possibility['test'])
                log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.Log_file)

                if last_min + 4 < new_min:

                    print('Saving clouds')

                    # Update last_min
                    last_min = new_min

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    files = dataset.test_files
                    i_test = 0
                    for i, file_path in enumerate(files):
                        # Get file
                        points = self.load_evaluation_points(file_path)
                        points = points.astype(np.float16)

                        # Reproject probs
                        probs = np.zeros(shape=[np.shape(points)[0], 4], dtype=np.float16)
                        proj_index = dataset.test_proj[i_test]

                        probs = self.test_probs[i_test][proj_index, :]

                        # Insert false columns for ignored labels
                        probs2 = probs

                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.uint8)

                        # Save plys
                        cloud_name = file_path.split('/')[-1]

                        # Save ascii preds
                        #ascii_name = join(test_path, 'predictions', dataset.ascii_files[cloud_name])
                        name = 'HUNHE' + '.txt'
                        print(name)
                        np.savetxt(join(test_path, 'predictions', name), preds, fmt='%d')
                        #np.savetxt(ascii_name, preds, fmt='%d')
                        #log_out(ascii_name + 'has saved', self.Log_file)
                        i_test += 1

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))
                    self.sess.close()
                    return

                self.sess.run(test_init_op)
                epoch_id += 1
                step_id = 0
                continue
        return
    @staticmethod
    def load_evaluation_points(file_path):
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T
def val():
    with tf.Graph().as_default():
        with tf.device('/gpu:2'):
            
           
            dataset =TONGJI()
            dataset.load_sub_sampled_clouds(FLAGS.first_subsampling_dl) #load_sub_sampled_clouds(FLAGS.first_subsampling_dl)
            #cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
            print('Initiating input pipelines')
            #ignored_label_inds = [dataset.label_to_idx[ign_label] for ign_label in dataset.ignored_labels]
            '''
            gen_function, gen_types, gen_shapes = dataset.get_batch_gen('training')
            gen_function_val, _, _ = dataset.get_batch_gen('validation')
            '''
            gen_function_test,gen_types, gen_shapes= dataset.get_batch_gen('test')
            '''
            train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
            val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
            '''
            test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
            '''
            batch_train_data = train_data.batch(FLAGS.batch_size)
            batch_val_data = val_data.batch(FLAGS.batch_size)
            '''
            batch_test_data = test_data.batch(FLAGS.batch_size)
            map_func = dataset.get_tf_mapping()
            '''
            batch_train_data = batch_train_data.map(map_func=map_func)
            batch_val_data = batch_val_data.map(map_func=map_func)
            '''
            batch_test_data = batch_test_data.map(map_func=map_func)
            '''
            batch_train_data = batch_train_data.prefetch(FLAGS.batch_size)
            batch_val_data = batch_val_data.prefetch(FLAGS.batch_size)
            '''
            batch_test_data = batch_test_data.prefetch(FLAGS.batch_size)

             # create a iterator of the correct shape and type
            iter = tf.data.Iterator.from_structure(batch_test_data.output_types, batch_test_data.output_shapes)
            flat_inputs = iter.get_next()
            
            # create the initialisation operations
            '''
            train_init_op = iter.make_initializer( batch_train_data)
            val_init_op = iter.make_initializer(batch_val_data)
            '''
            test_init_op = iter.make_initializer(batch_test_data)
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

            tester.test_cloud_segmentation_on_val(input, dataset, test_init_op)
          

if __name__ == "__main__":
    val()

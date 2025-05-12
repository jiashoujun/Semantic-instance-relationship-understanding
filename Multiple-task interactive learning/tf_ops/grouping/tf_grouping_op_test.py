import tensorflow as tf
import numpy as np
import os

from tf_grouping import query_ball_point, group_point
os.environ["CUDA_VISIBLE_DEVICES"]="2" 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True
sess = tf.Session(config=config)

class GroupPointTest(tf.test.TestCase):
  def test(self):
    pass

  def test_grad(self):
    with tf.device('/gpu:2'):
      points = tf.constant(np.random.random((1,128,16)).astype('float32'))
      print (points)
      xyz1 = tf.constant(np.random.random((1,128,3)).astype('float32'))
      xyz2 = tf.constant(np.random.random((1,8,3)).astype('float32'))
      radius = 0.3 
      nsample = 32
      idx, pts_cnt = query_ball_point(radius, nsample, xyz1, xyz2)
      grouped_points = group_point(points, idx)
      print (grouped_points)

    with self.test_session():
      print ("---- Going to compute gradient error")
      err = tf.test.compute_gradient_error(points, (1,128,16), grouped_points, (1,8,32,16))
      print (err)
      self.assertLess(err, 1e-4) 

if __name__=='__main__':
  tf.test.main() 

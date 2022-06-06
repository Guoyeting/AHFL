import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf
from models.cnn_abstract import ModelCNNAbstract

def lrn(x, radius = 5, alpha = 1e-04, beta = 0.75, bias = 2.0):
    return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
                                                beta = beta, bias = bias)

def max_pool(x, filter_height = 3, filter_width = 3, stride = 2, padding = 'VALID'):
    return tf.nn.max_pool(x, ksize = [1, filter_height, filter_width, 1],
                        strides = [1, stride, stride, 1], padding = padding)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob = keep_prob)

class ModelAlexNetCifar10(ModelCNNAbstract):
    def __init__(self):
        super().__init__()
        pass

    def create_graph(self, learning_rate=None):
        self.x = tf.placeholder(tf.float32, shape=[None, 3072])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        self.x_image = tf.reshape(self.x, [-1, 32, 32, 3])
        
        self.W_conv1 = tf.get_variable('weights_1', shape = [11, 11, 3, 96],
                                       initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        self.b_conv1 = tf.get_variable('biases_1', shape = [96],
                                       initializer=tf.constant_initializer(0.0))
        self.W_conv2 = tf.get_variable('weights_2', shape = [5, 5, 96/2, 256],
                                       initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        self.b_conv2 = tf.get_variable('biases_2', shape = [256],
                                       initializer=tf.constant_initializer(1.0))
        self.W_conv3 = tf.get_variable('weights_3', shape = [3, 3, 256, 384],
                                       initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        self.b_conv3 = tf.get_variable('biases_3', shape = [384],
                                       initializer=tf.constant_initializer(0.0))
        self.W_conv4 = tf.get_variable('weights_4', shape = [3, 3, 384/2, 256],
                                       initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        self.b_conv4 = tf.get_variable('biases_4', shape = [256],
                                       initializer=tf.constant_initializer(0.0))
        self.W_fc1 = tf.get_variable('weights_fc1', shape = [256, 1024],
                                    initializer = tf.random_normal_initializer(mean=0, stddev=0.01))
        self.b_fc1 = tf.get_variable('bias_fc1', shape = [1024],
                                    initializer = tf.constant_initializer(1.0))
        self.W_fc2 = tf.get_variable('weights_fc2', shape = [1024, 2048],
                                    initializer = tf.random_normal_initializer(mean=0, stddev=0.01))
        self.b_fc2 = tf.get_variable('bias_fc2', shape = [2048],
                                    initializer = tf.constant_initializer(1.0))
        self.W_fc3 = tf.get_variable('weights_fc3', shape = [2048, 10],
                                    initializer = tf.random_normal_initializer(mean=0, stddev=0.01))
        self.b_fc3 = tf.get_variable('bias_fc3', shape = [10],
                                    initializer = tf.constant_initializer(1.0))


        tmp = tf.nn.conv2d(self.x_image, self.W_conv1, strides = [1, 2, 2, 1], padding = 'SAME')
        self.h_conv1 = tf.nn.relu(tf.nn.bias_add(tmp, self.b_conv1))
        self.h_norm1 = lrn(self.h_conv1)
        self.h_pool1 = max_pool(self.h_norm1)

        input_groups = tf.split(axis = 3, num_or_size_splits = 2, value = self.h_pool1)
        weight_groups = tf.split(axis = 3, num_or_size_splits = 2, value = self.W_conv2)
        output_groups = [tf.nn.conv2d(i, k, strides=[1, 1, 1, 1], padding='SAME')
                         for i, k in zip(input_groups, weight_groups)]
        conv = tf.concat(axis=3, values=output_groups)
        self.h_conv2 = tf.nn.relu(tf.nn.bias_add(conv, self.b_conv2))
        self.h_norm2 = lrn(self.h_conv2)
        self.h_pool2 = max_pool(self.h_norm2)

        tmp = tf.nn.conv2d(self.h_pool2, self.W_conv3, strides = [1, 1, 1, 1], padding = 'SAME')
        self.h_conv3 = tf.nn.relu(tf.nn.bias_add(tmp, self.b_conv3))

        input_groups = tf.split(axis = 3, num_or_size_splits = 2, value = self.h_conv3)
        weight_groups = tf.split(axis = 3, num_or_size_splits = 2, value = self.W_conv4)
        output_groups = [tf.nn.conv2d(i, k, strides=[1, 1, 1, 1], padding='SAME')
                         for i, k in zip(input_groups, weight_groups)]
        conv = tf.concat(axis=3, values=output_groups)
        self.h_conv4 = tf.nn.relu(tf.nn.bias_add(conv, self.b_conv4))
        self.h_pool4 = max_pool(self.h_conv4)

        flattened = tf.reshape(self.h_pool4, [-1, 1*1*256])
        self.h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flattened, self.W_fc1), self.b_fc1))
        dropout5 = dropout(self.h_fc1, 0.5)

        self.h_fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dropout5, self.W_fc2), self.b_fc2))
        dropout6 = dropout(self.h_fc2, 0.5)

        self.y = tf.nn.bias_add(tf.matmul(dropout6, self.W_fc3), self.b_fc3)

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

        self.all_weights = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3,
                            self.W_conv4, self.b_conv4, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.W_fc3, self.b_fc3]

        self._assignment_init()

        self.learning_rate = 0.0001
        #self._optimizer_init(learning_rate=learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.grad = self.optimizer.compute_gradients(self.cross_entropy, var_list=self.all_weights)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.acc = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self._session_init()
        self.graph_created = True





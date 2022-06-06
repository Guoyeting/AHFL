import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
from models.cnn_abstract import ModelCNNAbstract

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

conv1_unit = 16
conv2_unit = 48

'''
def weight_variable(shape, clip_bound):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(tf.clip_by_norm(initial, clip_bound))

def bias_variable(shape, clip_bound):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(tf.clip_by_norm(initial, clip_bound))
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class ModelCNNMnist(ModelCNNAbstract):
    def __init__(self):
        super().__init__()
        pass

    def create_graph(self, learning_rate=None):
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        '''
        self.W_conv1 = weight_variable([5, 5, 1, 32], 2.6722343)
        self.b_conv1 = bias_variable([32], 0.58140683)
        self.W_conv2 = weight_variable([5, 5, 32, 32], 14.141903)
        self.b_conv2 = bias_variable([32], 0.582764)
        self.W_fc1 = weight_variable([7 * 7 * 32, 256], 55.756493)
        self.b_fc1 = bias_variable([256], 1.600285)
        self.W_fc2 = weight_variable([256, 10], 4.6593556)
        self.b_fc2 = bias_variable([10], 0.31674847)
        '''
        self.W_conv1 = weight_variable([5, 5, 1, conv1_unit])
        self.b_conv1 = bias_variable([conv1_unit])
        self.W_conv2 = weight_variable([5, 5, conv1_unit, conv2_unit])
        self.b_conv2 = bias_variable([conv2_unit])
        self.W_fc1 = weight_variable([7 * 7 * conv2_unit, 256])
        self.b_fc1 = bias_variable([256])
        self.W_fc2 = weight_variable([256, 10])
        self.b_fc2 = bias_variable([10])

        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)
        self.h_norm1 = tf.nn.lrn(self.h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        self.h_conv2 = tf.nn.relu(conv2d(self.h_norm1, self.W_conv2) + self.b_conv2)
        self.h_norm2 = tf.nn.lrn(self.h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        self.h_pool2 = max_pool_2x2(self.h_norm2)
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * conv2_unit])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        self.y = tf.nn.softmax(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

        self.all_weights = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2,
                            self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
        self.norm = []
        for item in self.all_weights:
            self.norm.append(tf.norm(item, ord=2))
        self._assignment_init()

        self._optimizer_init(learning_rate=learning_rate)
        self.grad = self.optimizer.compute_gradients(self.cross_entropy, var_list=self.all_weights)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.acc = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self._session_init()
        self.graph_created = True


    def create_former_graph(self, learning_rate, split_point):
        self.split_point = split_point
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        self.all_weights = []
        if split_point >= 1:
            #self.W_conv1 = weight_variable([5, 5, 1, 32],2.6722343)
            #self.b_conv1 = bias_variable([32], 0.58140683)
            self.W_conv1 = weight_variable([5, 5, 1, conv1_unit])
            self.b_conv1 = bias_variable([conv1_unit])
            self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
            self.h_pool1 = max_pool_2x2(self.h_conv1)
            self.h_norm1 = tf.nn.lrn(self.h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            self.all_weights.append(self.W_conv1)
            self.all_weights.append(self.b_conv1)
            if split_point >=2:
                self.W_conv2 = weight_variable([5, 5, conv1_unit, conv2_unit])
                self.b_conv2 = bias_variable([conv2_unit])
                self.h_conv2 = tf.nn.relu(conv2d(self.h_norm1, self.W_conv2) + self.b_conv2)
                self.h_norm2 = tf.nn.lrn(self.h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
                self.h_pool2 = max_pool_2x2(self.h_norm2)
                self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * conv2_unit])
                self.all_weights.append(self.W_conv2)
                self.all_weights.append(self.b_conv2)
                if split_point >= 3:
                    self.W_fc1 = weight_variable([7 * 7 * conv2_unit, 256])
                    self.b_fc1 = bias_variable([256])
                    self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
                    self.all_weights.append(self.W_fc1)
                    self.all_weights.append(self.b_fc1)


        self._assignment_init()
        self._optimizer_init(learning_rate=learning_rate)

        if split_point == 1:
            self.mid_grad = tf.placeholder(tf.float32, shape=[None, 14, 14, conv1_unit])
            self.grad = self.optimizer.compute_gradients(tf.math.reduce_sum(self.mid_grad * self.h_norm1), var_list=self.all_weights)
        elif split_point == 2:
            self.mid_grad = tf.placeholder(tf.float32, shape=[None, 7 * 7 * conv2_unit])
            self.grad = self.optimizer.compute_gradients(tf.math.reduce_sum(self.mid_grad * self.h_pool2_flat), var_list=self.all_weights)
        elif split_point == 3:
            self.mid_grad = tf.placeholder(tf.float32, shape=[None, 256])
            self.grad = self.optimizer.compute_gradients(tf.math.reduce_sum(self.mid_grad * self.h_fc1), var_list=self.all_weights)

        self._session_init()
        self.graph_created = True

    def create_latter_graph(self, learning_rate, split_point):
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        self.split_point = split_point
        if split_point == 1:
            self.h_norm1 = tf.placeholder(tf.float32, shape=[None, 14, 14, conv1_unit])
        elif split_point == 2:
            self.h_pool2_flat = tf.placeholder(tf.float32, shape=[None, 7 * 7 * conv2_unit])
        elif split_point == 3:
            self.h_fc1 = tf.placeholder(tf.float32, shape=[None, 256])

        if split_point <= 3:
            self.W_fc2 = weight_variable([256, 10])
            self.b_fc2 = bias_variable([10])
            if split_point <= 2:
                self.W_fc1 = weight_variable([7 * 7 * conv2_unit, 256])
                self.b_fc1 = bias_variable([256])
                if split_point <= 1:
                    self.W_conv2 = weight_variable([5, 5, conv1_unit, conv2_unit])
                    self.b_conv2 = bias_variable([conv2_unit])

        if split_point == 1:
            self.h_conv2 = tf.nn.relu(conv2d(self.h_norm1, self.W_conv2) + self.b_conv2)
            self.h_norm2 = tf.nn.lrn(self.h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            self.h_pool2 = max_pool_2x2(self.h_norm2)
            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * conv2_unit])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
            self.y = tf.nn.softmax(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)
            self.all_weights = [self.W_conv2, self.b_conv2, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
            self.all_weights_gradient = [self.W_conv2, self.b_conv2, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.h_norm1]
        elif split_point == 2:
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
            self.y = tf.nn.softmax(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)
            self.all_weights = [ self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
            self.all_weights_gradient = [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.h_pool2_flat]
        elif split_point == 3:
            self.y = tf.nn.softmax(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)
            self.all_weights = [self.W_fc2, self.b_fc2]
            self.all_weights_gradient = [self.W_fc2, self.b_fc2, self.h_fc1]


        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

        self._assignment_init()

        self._optimizer_init(learning_rate=learning_rate)
        self._optimizer_op_init(learning_rate=learning_rate)
        self.grad = self.optimizer.compute_gradients(self.cross_entropy, var_list=self.all_weights_gradient)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.acc = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self._session_init()
        self.graph_created = True
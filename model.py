import numpy as np
import tensorflow as tf
import logging
import tensorflow.contrib.slim as slim

logging.getLogger().setLevel(logging.INFO)

# Functions required by the class Model

''' 
Initializes a matrix of given shape with random numbers
'''


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


'''
Takes the input image, makes a weight variable and bias and then
applies ConvNet to it
'''


def conv(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1],
                        int(x.get_shape()[3]), num_filters]

        # For finding the bound for w
        num_hidden_in = np.prod(filter_shape[:3])
        num_hidden_out = np.prod(filter_shape[:2]) * num_filters
        w_bound = np.sqrt(6. / (num_hidden_in + num_hidden_out))

        # Getting the weights and bias
        w = tf.get_variable("W", filter_shape, dtype,
                            initializer=tf.random_uniform_initializer(-w_bound, w_bound))
        b = tf.get_variable("b", [1, 1, 1, num_filters],
                            initializer=tf.constant_initializer(0.0))

        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def fully_connected(x, output_size, name, initializer=None, bias_init=0):
    w = tf.get_variable(
        name + "/w", [x.get_shape()[1], output_size], initializer=initializer)
    b = tf.get_variable(
        name + "/b", [output_size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

# Actual class Model - the main class for this file that calculates the action
# and the state value given an image screenshot


class Model():
    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(
            shape=[None] + list(ob_space), dtype=tf.float32)

        # Using 4 Convolutional layers
        for i in range(4):
            x = tf.nn.elu(conv(x, 32, "l" + str(i), stride=[2, 2]))

        self.after_conv = x

        hidden = slim.fully_connected(
                slim.flatten(x), 256, activation_fn=tf.nn.elu)
        # Reshaping into 256 nodes
        x = tf.reshape(hidden, [-1, 256])

        self.policy = fully_connected(
            x, ac_space, "action", normalized_columns_initializer(0.01))
        epsila_greedy = tf.multinomial(self.policy, 1)[0]
        self.action = tf.one_hot(epsila_greedy, ac_space)
        self.value = tf.reshape(fully_connected(
            x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name) 

    def act(self, state):
        sess = tf.get_default_session()
        action_onehot, value = sess.run(
            [self.action, self.value], feed_dict={self.x: [state]})

        return action_onehot, value

    def val(self, state):
        sess = tf.get_default_session()
        val = sess.run(self.value, feed_dict={self.x: [state]})[0]
        return val
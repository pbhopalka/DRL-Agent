import numpy as np
import tensorflow as tf
import logging

logging.getLogger().setLevel(logging.INFO)

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def conv(x, output_size, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float.32):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), output_size]

        # For finding the bound for w
        num_hidden_in = np.prod(filter_shape[:3])
        num_hidden_out = np.prod(filter_shape[:2]) * output_size
        w_bound = np.sqrt(6. / (num_hidden_in + num_hidden_out))

        # Getting the weights and bias
        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound))
        b = tf.get_variable("b", [1, 1, 1, output_size], initializer=tf.constant_initializer(0.0))

        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def fully_connected(x, output_size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], output_size], initializer=initializer)
    b = tf.get_variable(name + "/b", [output_size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

class Model():
    def __init__(self, ob_space, ac_space):
        self.x = tf.placeholder(shape=[None] + list(ob_space), dtype = tf.float32)

        # Using 4 Convolutional layers
        for i in range(4):
            x = tf.nn.elu(conv(x, ac_space, "l"+str(i), stride=[2,2]))
        
        # Reshaping into 256 nodes
        x = tf.reshape(x, [-1, 256])

        self.policy = fully_connected(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.action = tf.one_hot(self.policy, ac_space)
        self.value = tf.reshape(fully_connected(x, ac_space, "value", normalized_columns_initializer(1.0)), [-1])
    
    def act(self, state):
        sess = tf.get_default_session()
        action_onehot, value = sess.run([self.action, self.value], {self.x: [state]})
        
        logging.info("Action:")
        logging.info(action_onehot)
        logging.info(value)

        return action_onehot, value

    def value(self, state):
        sess = tf.get_default_session()
        val = sess.run(self.value, {self.x: [state]})[0]
        return val
'''
This page contains the code for building the model. 
The model here consists of (Input Image) -> CNN -> LSTM -> State Value
                                                    |
                                                    V
                                                Action-value

The value of State-Value V(s) and Action-Value Q(s, a) are then sent to the agent
to make changes in the environment.BaseException

- Team Convolution
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import logging
import tensorflow.contrib.slim as slim

logging.getLogger().setLevel(logging.INFO)

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
def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

'''
Calls the tf.nn.conv2d function by initializing filter, if not already done
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

'''
A simple code for a fully connected layer without any kind of activation function
'''
def fully_connected(x, output_size, name, initializer=None, bias_init=0):
    w = tf.get_variable(
        name + "/w", [x.get_shape()[1], output_size], initializer=initializer)
    b = tf.get_variable(
        name + "/b", [output_size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

'''
The main Model Class that defines the model
It has two functions that generate the action and values given a state,
cell-state (c) and history (h) as inputs.
c and h are required for LSTM
'''
class Model():
    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(
            shape=[None] + list(ob_space), dtype=tf.float32)

        # Using 4 Convolutional layers
        for i in range(4):
                x = tf.nn.elu(conv(x, 32, "l" + str(i), stride=[2, 2]))

        x = tf.expand_dims(flatten(x), [0])

        size = 256
        lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        # print "LSTM state size:", lstm.state_size

        step_size = tf.shape(self.x)[:1]
        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        
        self.c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        self.h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        # self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(self.c_in, self.h_in)

        output, l_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size)
        c_out, h_out = l_state
        # Reshaping into 256 nodes
        x = tf.reshape(output, [-1, size])
        self.state_out = [c_out[:1, :], h_out[:1, :]]

        
        self.policy = fully_connected(
            x, ac_space, "action", normalized_columns_initializer(0.01))
        epsila_greedy = tf.multinomial(self.policy, 1)[0]
        self.action = tf.one_hot(epsila_greedy, ac_space)
        self.value = tf.reshape(fully_connected(
            x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name) 

    def get_init_features(self):
        return self.state_init
    
    def act(self, state, c, h):
        # print "After acting: c: ", features[0].size
        # print "After acting, h: ", features[1].size
        sess = tf.get_default_session()
        action_onehot, value, feature = sess.run(
            [self.action, self.value, self.state_out], 
                feed_dict={self.x: [state], self.c_in: c, self.h_in: h})

        return action_onehot, value, feature

    def val(self, state, c, h):
        sess = tf.get_default_session()
        val = sess.run(self.value, 
                feed_dict={self.x: [state], self.c_in: c, self.h_in: h})[0]
        return val
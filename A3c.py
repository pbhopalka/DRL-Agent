import numpy as np
import tensorflow as tf
import random

from model import Model
from game import RunThread, process_buffer

class A3C():
    def __init__(self, env, number):
        self.env = env
        self.number = number

        worker_device = "/job:worker/task:"+str(number)+"/cpu:0"
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = Model(env.observation_space.shape, env.action_space.n)
                self.global_step = tf.get_variable("global_step", [], tf.int32, 
                    initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_AC = Model(env.observation_space.shape, env.action_space.n)
                self.local_AC.global_step = self.global_step

            self.action = tf.placeholder(tf.float32, [None, env.action_space.n])
            self.advantage = tf.placeholder(tf.float32, [None])
            self.target_v = tf.placeholder(tf.float32, [None])

            log_prob = tf.nn.log_softmax(self.local_AC.policy)
            prob = tf.nn.softmax(self.local_AC.policy)

            responsible_output = tf.reduce_sum(log_prob * self.action, [1])
            policy_loss = - tf.reduce_sum(responsible_output * self.advantage)

            value_loss = 0.5 * tf.reduce_sum(tf.square(self.local_AC.value - self.target_v))
            entropy = -tf.reduce_sum(prob * log_prob)

            base = tf.to_float(tf.shape(self.local_AC.x)[0])
            self.loss = 0.5 * value_loss + policy_loss - entropy * 0.01 

            self.runner = RunThread(env, self.local_AC, 20)

            grads = tf.gradients(self.loss, self.local_AC.var_list)

            tf.summary.scalar("Model/policy_loss", policy_loss/base)
            tf.summary.scalar("Model/value_loss", value_loss/base)
            tf.summary.scalar("Model/entropy", entropy/base)
            tf.summary.image("Model/Image_Input", self.local_AC.x)
            # tf.summary.image("Model/After_conv", self.local_AC.after_conv)
            tf.summary.scalar("Model/Gradient_Global_norm", tf.global_norm(grads))
            self.summary_op = tf.summary.merge_all()

            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            self.update_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.local_AC.var_list, self.network.var_list)])

            grads_vars = list(zip(grads, self.network.var_list))
            inc_global_step = self.global_step.assign_add(tf.shape(self.local_AC.x)[0])

            trainer = tf.train.RMSPropOptimizer(1e-4)
            self.train_op = tf.group(trainer.apply_gradients(grads_vars), inc_global_step)

    def start_app(self, sess, summary_writer):
        self.runner.start_thread(sess, summary_writer)
        self.summary_writer = summary_writer

    def get_buffer_from_queue(self):
        buffer_list = self.runner.queue.get(timeout=600.0)
    
    def train_data(self, sess):
        sess.run(self.update_op)
        batches = self.runner.queue.get(timeout=600.0)
        state, action, reward, advantage = process_buffer(batches, gamma=0.99, lamda=1.0)

        fetches = [self.summary_op, self.train_op, self.global_step]

        feed_dict = {
            self.local_AC.x: state,
            self.action:action,
            self.advantage:advantage, 
            self.target_v:reward
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)
        
        self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
        self.summary_writer.flush()

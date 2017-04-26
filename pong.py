import numpy as np
import os
import threading
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import multiprocessing
import scipy.signal

from random import choice
from time import sleep
from time import time

from helper import *
from model import *
from A3c import A3C

import logging

'''
    Use logging:
    logging.getLogger().setLevel(logging.INFO)
    Then use logging.info
'''
# For logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# class A3C(object):
#     def __init__(self, ob_space, ac_space, scope, trainer):
#         with tf.variable_scope(scope):
#             self.network = Model(ob_space,ac_space)
            
#             if scope != 'global':
#                 # setting 6 as the env.action_space.n
#                 self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
#                 self.action_onehot = tf.one_hot(self.actions, ac_space, dtype=tf.float32)
#                 self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
#                 self.advantage = tf.placeholder(shape=[None], dtype=tf.float32)

#                 self.log_policy = tf.nn.log_softmax(self.network.policy)
#                 self.policy = tf.nn.softmax(self.network.policy)

#                 self.responsible_output = tf.reduce_sum(
#                     self.log_policy * self.action_onehot, [1])

#                 self.value_loss = 0.5 * tf.reduce_mean(
#                     tf.square(self.target_v - self.network.value))
#                 self.entropy = - tf.reduce_sum(
#                     self.policy * self.log_policy)
#                 self.policy_loss = -tf.reduce_sum(
#                     self.responsible_output * self.advantage)
#                 self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

#                 #Get gradient from local network using local losses
#                 local_vars = tf.get_collection(
#                     tf.GraphKeys.TRAINABLE_VARIABLES, scope)
#                 self.gradients = tf.gradients(self.loss, local_vars)
#                 self.var_norms = tf.global_norm(local_vars)
#                 grads, self.grad_norms = tf.clip_by_global_norm(
#                     self.gradients, 40.0)

#                 #Apply local gradient to global network
#                 global_vars = tf.get_collection(
#                     tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
#                 self.apply_grads = trainer.apply_gradients(
#                     zip(grads, global_vars))


class Worker(object):
    #Assuming that game is already initialized and sent in env
    def __init__(self, env, name, ob_space, ac_space, model_path):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        # self.trainer = trainer
        # self.global_episodes = global_episodes
        # if more than 1 image is sent, then this needs to be implemented accordingly
        # self.increment = self.global_episodes.assign_add(1) 
        self.episode_reward = []
        self.episode_length = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" +
                                                    str(self.number))

        # print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

        # self.local_AC = A3C(ob_space, ac_space, self.name, trainer)
        self.actor_critic = A3C(env, self.number)
        # self.update_local_op = update_target_graph('global', self.name)

        self.env = env

    def train(self, rollout, sess, gamma, value_state):
        # rollout = np.asarray(rollout)
        observation = np.asarray([item[0] for item in rollout])
        # rollout = np.asarray(rollout)
        actions = np.asarray([item[1] for item in rollout])
        rewards = np.asarray([item[2] for item in rollout])
        next_observation = np.asarray([item[3] for item in rollout])
        values = np.asarray([item[5] for item in rollout])

        #Take rewards and values from rollout and generate advantage and discounted rewards
        #TODO: Remind yourself this works because rewards is already a nparray
        self.rewards_plus = np.asarray(rewards + [value_state]) 
        discounted_reward = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values + [value_state])
        advantage = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantage = discount(advantage, gamma)

        #Update global network using gradients from loss
        #Generate data to save data periodically
        feed_dict = {
            self.actor_critic.target_v: discounted_reward,
            self.actor_critic.local_AC.x: observation,
            self.actor_critic.action: actions,
            self.actor_critic.advantage: advantage
        }
        fetched = sess.run(
            [
                self.actor_critic.summary_op, self.actor_critic.train_op,
                self.actor_critic.global_step
            ],
            feed_dict=feed_dict)
        self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
        self.summary_writer.flush()

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker: ", str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_op)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0

                observation = self.env.reset()
                episode_frames.append(observation)
                #observation = process_frame(observation)

                done = False
                while not done:
                    action_vector, value = self.actor_critic.local_AC.act(observation)
                    
                    new_observation, reward, done, info = self.env.step(
                        action_vector.argmax())
                    
                    if self.number == 0:
                        self.env.render()
                    
                    # episode_frames.append(new_observation)

                    episode_buffer.append([
                        observation, action_vector.argmax(), reward,
                        new_observation, done, value[0]
                    ])
                    # episode_values.append(value[0])

                    episode_reward += reward
                    total_steps += 1
                    episode_step_count += 1

                    observation = new_observation

                    # Not necessary: For pong because no of lives left is what is shown
                    # if info:
                    #     summ = tf.Summary()
                    #     for k, v in info.items():
                    #         summ.value.add(tag="Info/"+k, simple_value=float(v))
                    #     self.summary_writer.add_summary(summ, episode_step_count)

                    # After every 20 steps in an episode, we train and 
                    # our local parameters are updated
                    if len(episode_buffer) == 20 and not done:
                        #bootstrapping the value function of the last obtained state
                        value_state = self.actor_critic.local_AC.val(observation)

                        sess.run(self.actor_critic.update_op)
                        self.train(episode_buffer, sess, gamma, value_state)
                        #Made episode buffer empty here. Don't know why
                        episode_buffer = []
                        
                    
                    # episode_step_count is the total number of frames that is taken as inputs
                    if episode_step_count >= max_episode_length - 1 or done:
                        logging.info("Episode count: %d, episode reward: %d", episode_count, episode_reward)
                        if not done:
                            logging.info("Terminated after episode step count - %d", episode_step_count)
                        break

                self.episode_reward.append(episode_reward)
                self.episode_length.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # for the last set of episodes that wasn't trained because it reached
                # terminal state
                if len(episode_buffer) != 0:
                    self.train(episode_buffer, sess, gamma, 0.0)

                #all secondary stuff like creating gifs
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' +
                                   str(episode_count) + '.cptk')
                        print "Saved Model"

                    mean_reward = np.mean(self.episode_reward[-5:])
                    mean_length = np.mean(self.episode_length[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])

                    summary = tf.Summary()

                    summary.value.add(
                        tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(
                        tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(
                        tag='Perf/Value', simple_value=float(mean_value))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()

                # if self.name == 'worker_0':
                # global_episode_number = sess.run(self.increment)
                episode_count += 1


max_episode_length = 1000000
gamma = 0.99
load_model = False
model_path = './model'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
env = gym.make("Pong-v0")
s_size = env.observation_space.shape
a_size = env.action_space.n

# with tf.device("/cpu:0"):
# global_episodes = tf.Variable(
#     0, dtype=tf.int32, name="global_episodes", trainable=False)
# global_episodes = tf.get_variable("global_episode", [], tf.int32, 
#     tf.constant_initializer(0, tf.int32), trainable=False)
# trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
# master_network = A3C(s_size, a_size, 'global', None)
num_workers = multiprocessing.cpu_count()
# num_workers = 2
workers = []

for i in range(num_workers):
    env_name = "env" + str(i)
    env_name = gym.make("Pong-v0")
    workers.append(
        Worker(env_name, i, s_size, a_size, model_path))

saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    # Printing all the variables that exist in this entire graph
    all_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        tf.get_variable_scope().name)
    logger.info("Trainable variable list:")
    for variable in all_var_list:
        logger.info(" %s %s", variable.name, variable.get_shape())
    
    if load_model:
        print "Loading Model..."
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    
    worker_thread = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.setDaemon(True)
        t.start()
        sleep(0.5)
        worker_thread.append(t)
    coord.join(worker_thread)

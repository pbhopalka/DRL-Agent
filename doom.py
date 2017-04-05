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


class A3C(object):
    def __init__(self, ob_space, ac_space, scope, trainer):
        self.inputs = tf.placeholder(shape=[None] + list(ob_space), dtype=tf.float32)

        self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])

        self.conv1 = slim.conv2d(
            activation_fn=tf.nn.elu,
            inputs=self.imageIn,
            num_outputs=16,
            kernel_size=[8, 8],
            stride=[4, 4],
            padding="VALID")

        self.conv2 = slim.conv2d(
            activation_fn=tf.nn.elu,
            inputs=self.conv1,
            num_outputs=32,
            kernel_size=[4, 4],
            stride=[2, 2],
            padding="VALID")

        hidden = slim.fully_connected(
            slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu)

        self.policy = slim.fully_connected(
            hidden,
            ac_space,
            activation_fn=tf.nn.softmax,
            weights_initializer=normalized_columns_initializer(0.01),
            biases_initializer=None)

        self.value = slim.fully_connected(
            hidden,
            1,
            activation_fn=None,
            weights_initializer=normalized_columns_initializer(1.0),
            biases_initializer=None)

        if scope != 'global':
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(
                self.actions, ac_space, dtype=tf.float32)
            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantage = tf.placeholder(shape=[None], dtype=tf.float32)

            self.responsible_output = tf.reduce_sum(
                self.policy * self.actions_onehot, [1])

            self.value_loss = 0.5 * tf.reduce_mean(
                tf.square(self.target_v - tf.reshape(self.value, [-1])))
            self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
            self.policy_loss = -tf.reduce_sum(
                tf.log(self.responsible_output) * self.advantage)
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

            #Get gradient from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(
                self.gradients, 40.0)

            #Apply local gradient to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


class Worker(object):
    #Assuming that game is already initialized and sent in env
    def __init__(self, env, name, ob_space, ac_space, trainer, model_path,
                 global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_reward = []
        self.episode_length = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" +
                                                    str(self.number))
        
        # print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

        self.local_AC = A3C(ob_space, ac_space, self.name, trainer)
        self.update_local_op = update_target_graph('global', self.name)

        self.env = env

    def train(self, rollout, sess, gamma, value_state):
        rollout = np.array(rollout)
        observation = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observation = rollout[:, 3]
        values = rollout[:, 5]

        #Take rewards and values from rollout and generate advantage and discounted rewards
        self.rewards_plus = np.asarray(rewards.tolist() + [value_state])
        discounted_reward = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist(), [value_state])
        advantage = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantage = discount(advantage, gamma)

        #Update global network using gradients from loss
        #Generate data to save data periodically
        feed_dict = {
            self.local_AC.target_v: discounted_reward,
            self.local_AC.inputs: np.vstack(observation),
            self.local_AC.actions: actions,
            self.local_AC.advantage: advantage,
        }
        value_loss, policy_loss, entropy_loss, grad_norm, var_norm = sess.run(
            [
                self.local_AC.value_loss, self.local_AC.policy_loss,
                self.local_AC.entropy, self.local_AC.grad_norms,
                self.local_AC.var_norms
            ],
            feed_dict=feed_dict)
        return value_loss / len(rollout), policy_loss / len(
            rollout), entropy_loss / len(rollout), grad_norm, var_norm

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
                d = False

                observation = self.env.reset()
                episode_frames.append(observation)
                observation = process_frame(observation)

                while True:
                    self.env.render()
                    action_vector, value = sess.run(
                        [self.local_AC.policy, self.local_AC.value],
                        feed_dict={self.local_AC.inputs: [observation]})

                    print "Value obtained: ", value

                    new_observation, reward, done, info = env.step(
                        action_vector.argmax())

                    episode_buffer.append([
                        observation, action_vector.argmax(), reward,
                        new_observation, done, value[0, 0]
                    ])
                    episode_values.append(value[0, 0])

                    episode_reward += reward
                    total_steps += 1
                    episode_step_count += 1

                    observation = new_observation
                    episode_frames.append(observation)
                    observation = process_frame(observation)

                    #If episode hasn't ended but experience replay is full, 
                    #then we update using experience rollout
                    if len(
                            episode_buffer
                    ) == 30 and not done and episode_step_count != max_episode_length - 1:
                        #bootstrapping the value function of the last obtained state
                        value_state = sess.run(
                            [self.local_AC.value],
                            feed_dict={self.local_AC.inputs:
                                       observation})[0, 0]
                        value_loss, policy_loss, entropy_loss, grad_norm, var_norm = self.train(
                            episode_buffer, sess, gamma, value_state)
                        sess.run(self.update_local_op)

                    if done:
                        break

                self.episode_reward.append(episode_reward)
                self.episode_length.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if len(episode_buffer) != 0:
                    value_loss, policy_loss, entropy_loss, grad_norm, var_norm = self.train(
                        episode_buffer, sess, gamma, 0.0)

                #all secondary stuff like creating gifs
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(
                            images,
                            './frames/image' + str(episode_count) + '.gif',
                            duration=len(images) * time_per_step,
                            true_image=True,
                            salience=False)
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
                    summary.value.add(
                        tag='Losses/Value Loss',
                        simple_value=float(value_loss))
                    summary.value.add(
                        tag='Losses/Policy Loss',
                        simple_value=float(policy_loss))
                    summary.value.add(
                        tag='Losses/Entropy', simple_value=float(entropy_loss))
                    summary.value.add(
                        tag='Losses/Grad Norm', simple_value=float(grad_norm))
                    summary.value.add(
                        tag='Losses/Var Norm', simple_value=float(var_norm))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)

                episode_count += 1


max_episode_length = 300
gamma = 0.99
load_model = False
model_path = './model'

env = gym.make("Pong-v0")

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(
        0, dtype=tf.int32, name="global_episodes", trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = A3C(env.observation_space.shape, env.action_space.n, 'global', None)
    num_workers = multiprocessing.cpu_count()
    workers = []

    for i in range(num_workers):
        workers.append(
            Worker(env, i, env.observation_space.shape, env.action_space.n, trainer, model_path, global_episodes))

    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()

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
        t.start()
        sleep(0.5)
        worker_thread.append(t)
    coord.join(worker_thread)

import numpy as np
import scipy
import tensorflow as tf
import threading
import Queue

from helper import discount

'''
This class stores the experiences of the game in the form
(state, action, reward, terminal, next_state, features(c and h))
'''
class ReplayBuffer(object): #for storing the experience replays

    def __init__(self, random_seed=123):
        self.count = 0
        self.buffer = []
        self.value_state = 0

    def add(self, s, a, r, t, s2, feature): #state, action, reward, values, new state, c, h
        experience = (s, a, r, t, s2, feature)
        self.buffer.append(experience)
        self.count += 1

    def add_value_state(self, value):
        self.value_state = value

    def size(self):
        return self.count

    def sample_batch(self):
        '''
        give the sample batch from replay buffer. If buffer_size < batch_size,
        return all the elements of buffer. Generally, we wait for buffer size
        to be atleast the size of batch_size
        '''
        batch = self.buffer

        s_batch = np.asarray([_[0] for _ in batch])
        a_batch = np.asarray([_[1] for _ in batch])
        r_batch = np.asarray([_[2] for _ in batch])
        t_batch = np.asarray([_[3] for _ in batch])
        s2_batch = np.asarray([_[4] for _ in batch])
        features = [_[5] for _ in batch][0]

        return s_batch, a_batch, r_batch, t_batch, features, self.value_state

    def clear(self):
        self.buffer = []
        self.count = 0

'''
This class instantiates from the Thread class of python.
It creates a new thread for each worker that we need.
'''
class RunThread(threading.Thread):
    def __init__(self, env, policy, number_local_steps):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(5)
        self.env = env
        self.policy = policy
        self.features = None
        self.number_local_steps = number_local_steps
        self.daemon = True
        self.sess = None
        self.summary_writer = None

    def start_thread(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            buffer_list = run_game_instance(self.env, self.policy, self.number_local_steps, self.summary_writer)
            while True:
                self.queue.put(next(buffer_list), timeout=600.0)


'''
Runs a game instance. 
'''
def run_game_instance(env, policy, number_local_steps, summary_writer):

    episode_values = []
    episode_reward = 0
    episode_step_count = 0 #length

    observation = env.reset()
    feature = policy.state_init
    print "feature", feature
    while True:
        buffer_list = ReplayBuffer()

        for _ in range(number_local_steps):
            action_vector, value, new_feature = policy.act(observation, *feature)
            new_observation, reward, done, info = env.step(action_vector.argmax())

            env.render()
            buffer_list.add(observation, action_vector[0], reward, value[0], new_observation, feature)
            episode_values.append(value[0])
            episode_reward += reward
            episode_step_count += 1

            observation = new_observation
            feature = new_feature

            # Not necessary: For pong because no of lives left is what is shown
            if info:
                summ = tf.Summary()
                for k, v in info.items():
                    summ.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summ, policy.global_step.eval())
                summary_writer.flush()

            env_time_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if done or episode_step_count >= env_time_limit:
                if episode_step_count >= env_time_limit or not env.metadata.get('semantics.autoreset'):
                    observation = env.reset()
                feature = policy.state_init
                print ("Episode over. Rewards: %d, Length: %d" %(episode_reward, episode_step_count))

                episode_step_count = 0
                episode_reward = 0
                break

        if not done:
            value_state = policy.val(observation, *feature)
            # print "Value state: ", value_state
            buffer_list.add_value_state(value_state)

        yield buffer_list

'''
Process the buffer and calculate the discounted rewards and advantge
'''
def process_buffer(buffer_list, gamma, lamda=1.0):
    fetched = buffer_list.sample_batch()
    batch_si, action, rewards, values, features, value_state = fetched[0], fetched[1], fetched[2], fetched[3], fetched[4], fetched[5]
    
    #Take rewards and values from buffer_list and generate advantage and discounted rewards
    rewards_plus = np.append(rewards, [value_state]) 
    discounted_reward = discount(rewards_plus, gamma)[:-1]
    value_plus = np.append(values, [value_state])
    advantage = rewards + gamma * value_plus[1:] - value_plus[:-1]
    advantage = discount(advantage, gamma * lamda)

    return batch_si, action, discounted_reward, advantage, features

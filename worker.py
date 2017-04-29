import sys
import signal
import go_vncdriver
import tensorflow as tf
import time
import argparse

from envs import create_atari_env
from A3c import A3C

def run_game(env_name, number, server):
    env = create_atari_env(env_name)


    logdir = '/tmp/' + env_name + '-adam-LSTM/'
    # Try using Adam, RMSProp, Adadelta, etc.
    trainer = tf.train.AdamOptimizer(1e-4)

    actor_critic = A3C(env, number, trainer)

    var_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
    init_op = tf.variables_initializer(var_to_save)
    init_all_op = tf.global_variables_initializer()

    def init_fn(sess):
        # logger.info("Intializing parameters")
        sess.run(init_all_op)

    saver = tf.train.Saver(max_to_keep=5)
    save_path = logdir + "model/"
    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:"+str(number)+"/cpu:0"])
    summary_writer = tf.summary.FileWriter(logdir + "train_"+str(number))

    # Using tf.train.Supervisor
    # TODO: Expanded version of this
    is_chief = (number == 0)
    sv = tf.train.Supervisor(is_chief=is_chief, logdir=logdir, saver=saver, summary_op=None, init_op=init_op, init_fn=init_fn, summary_writer=summary_writer,
                            ready_op=tf.report_uninitialized_variables(var_to_save), global_step=actor_critic.global_step, 
                            save_model_secs=30, save_summaries_secs=30)

    num_global_steps = 100000000

    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        sess.run(actor_critic.update_op)
        # TODO:Code for running the program (As of now, inside pong.py, its Worker class)
        actor_critic.start_app(sess, summary_writer)
        global_step = sess.run(actor_critic.global_step)
        tf.logging.info("Starting training at step=%d", global_step)

        while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
            # TODO:Code for doing the training work. Inside pong.py, Worker.train()
            actor_critic.train_data(sess)
            global_step = sess.run(actor_critic.global_step)

    sv.stop()
    tf.logging.info("Reached %s steps. Worker stopped", global_step)
    
def parallel_work(num_workers, num_ps):
    # For using Distributed Tensorflow
    # Defining the cluster ports for ps(parameter server) and workers
    cluster = {}
    port = 12222

    ps = []
    host = '127.0.0.1:'
    for i in range(num_ps):
        ps.append(str(host)+str(port))
        port += 1
    cluster['ps'] = ps

    worker = []
    for i in range(num_workers):
        worker.append(str(host)+str(port))
        port += 1
    cluster['worker'] = worker
    print cluster
    return cluster

def main():
    tf.logging.set_verbosity(3)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default=0, type=int)    
    parser.add_argument('--env-id', default="Pong-v0")
    parser.add_argument('--num-workers', default=1, type=int)
    parser.add_argument('--job-name', default="worker")

    arguments = parser.parse_args()

    spec = parallel_work(arguments.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        # TODO: Find out what the argument means
        sys.exit(128+signal)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if arguments.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=arguments.task, 
                            config=tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2))
        run_game(arguments.env_id, arguments.task, server)

    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=arguments.task, 
                            config=tf.ConfigProto(device_filters=["/job:ps"]))
        print "After server in ps"
        while True:
            time.sleep(1000)


main()

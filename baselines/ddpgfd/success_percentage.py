import matplotlib.pyplot as plt
from baselines import logger
import pandas
import argparse
import numpy as np
from matplotlib.pyplot import cm

import solveHMS.envs
import gym
import tensorflow as tf
import pickle

from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import baselines.common.tf_util as U

from baselines import logger
import numpy as np




def success_percentage(file_path, **kwargs):

    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)

        vals = pickle.load(open(model_name, 'rb'))
        var = tf.trainable_variables()
        for v in var:
            assign_op = v.assign(vals[v.name])
            sess.run(assign_op)

        agent.sess = sess

        for i_rollout in range(10):
            print('rollout_no: ', i_rollout)

            rewards = []
            observations0 = []
            actions = []
            observations1 = []
            terminals = []

            obs = env.reset()
            episode_rew = 0
            done = False

            while not done:
                env.render()
                #action = act(obs[None, :])
                observations0.append(obs)
                action, q = agent.pi(obs, apply_noise=False)

                assert max_action.shape == action.shape
                new_obs, r, done, info = env.step(max_action * action)
                actions.append(action)
                observations1.append(obs)
                terminals.append(done)
                rewards.append(r)

                obs = new_obs

            print('reward', np.sum(np.asarray(rewards)))

            expert_data = {'observations': np.array(observations0),
                           'actions': np.array(actions),
                           'rewards': np.array(rewards)}
            experts.append(expert_data)

        np.save(saving_folder + 'demo-collected-after-loading-trained-policy-'+time.strftime('%Y-%m-%d-%H-%M-%S', current_time)+'.npy', experts)

        success_percentage = env.evaluate_success(experts)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file-path', type=str, default='trained_variable-DDPG.ckpt')
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameter
    dict_args = vars(args)
    return dict_args

if __name__ == "__main__":
    args = parse_args()
    plot(**args)

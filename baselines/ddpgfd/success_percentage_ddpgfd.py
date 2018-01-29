import matplotlib.pyplot as plt
from baselines import logger
import pandas
import argparse
import numpy as np
from matplotlib.pyplot import cm

import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.ddpgfd.training as training
from baselines.ddpgfd.models import Actor, Critic
from baselines.ddpgfd.memory import Memory
from baselines.ddpgfd.noise import *



import gym
import solveHMS.envs
import tensorflow as tf
import pickle

from baselines.ddpgfd.ddpgfd import DDPGFD
from baselines.ddpgfd.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import glob

def success_perc(env_id, seed, noise_type, layer_norm, evaluation, execution,model_name, saving_folder,nb_min_demo,  **kwargs):

    e = gym.make(env_id)
    nb_actions = e.action_space.shape[-1]
    max_action = e.action_space.high
    demonstrations = []
    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=e.action_space.shape, observation_shape=e.observation_space.shape, nb_min_demo=nb_min_demo, demonstrations=demonstrations, eps_d=0.0)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)


    tf.reset_default_graph()
    set_global_seeds(seed)
    #e.seed(seed)


    agent = DDPGFD(actor, critic, memory, e.observation_space.shape, e.action_space.shape)
        #gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        ##batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        #actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        #reward_scale=reward_scale)

    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)

        agent.sess = sess

        count_model = -1

        num_models = len(os.listdir(os.getcwd()))

        successes = np.zeros(num_models)

        for model_name in sorted(os.listdir(os.getcwd())):

            #agent.initialize(sess)
            #agent.reset()
            print('model_name', model_name)
            vals = pickle.load(open(model_name, 'rb'))
            var = tf.trainable_variables()
            for v in var:
                assign_op = v.assign(vals[v.name])
                agent.sess.run(assign_op)


            experts = []
            current_time = time.localtime()
            #obs = e.reset()

            for i_rollout in range(3):
                #print('rollout_no: ', i_rollout)

                rewards = []
                observations0 = []
                actions = []
                observations1 = []
                terminals = []

                obs = e.reset()
                #agent.reset()
                episode_rew = 0
                done = False

                while not done:
                    #e.render()
                    #action = act(obs[None, :])
                    observations0.append(obs)

                    action, q = agent.pi(obs, apply_noise=False)

                    new_obs, r, done, info = e.step( max_action * action)
                    actions.append(action)
                    observations1.append(obs)
                    terminals.append(done)

                    obs = new_obs

                    rewards.append(r)

                    #agent.store_transition(obs, actions, rewards, observations1, terminals, execute=True)

                print('reward', np.sum(np.asarray(rewards)))

                expert_data = {'observations': np.array(observations0),
                               'actions': np.array(actions),
                               'rewards': np.array(rewards)}
                experts.append(expert_data)

                np.save(saving_folder + 'demo-collected-after-loading-trained-policy-Pen'+time.strftime('%Y-%m-%d-%H-%M-%S', current_time)+'-'+str(count_model)+'.npy', experts)

                success_percentage = e.env.evaluate_success(experts)

                print(success_percentage)

                print(count_model)

            count_model += 1

            #if count_model == 51:

            successes[count_model]=success_percentage

    x = np.arange(0, num_models)
    y = successes
    plt.xlabel('Epochs ', fontsize=8)
    plt.ylabel('Success Percentage', fontsize=8)
    plt.plot(x,y, linewidth=2, c='b')

    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='pen_reposition-v0')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--batch-size-bc', type=int, default=64)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=3)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    parser.add_argument('--demo-file', type=str, default='demo-collected-2.npy') # minimum number of demo guaranteed in the replay buffer
    parser.add_argument('--alpha', type=float, default=0.3) # alpha value for priorization
    parser.add_argument('--eps', type=float, default=0.005) # constant for priorization computation
    parser.add_argument('--eps_d', type=float, default=0.01) # constant for priorization computation
    parser.add_argument('--lambda-3', type=float, default=1.0) # weight for priorization computation
    parser.add_argument('--target-period-update', type=int, default=20) # target networks are updated every target_period_update training steps
    parser.add_argument('--nb-training-bc', type=int, default=10000) # number of behaviour_cloning training step to be performed
    parser.add_argument('--t-inner-steps', type=int, default=1)
    parser.add_argument('--n-value', type=int, default=5)
    boolean_flag(parser, 'evaluation', default=False)
    parser.add_argument('--lambda-n', type=float, default=0.1) # weight for priorization computation
    boolean_flag(parser, 'behaviour-cloning-off', default=False)
    boolean_flag(parser, 'priorization-off', default=False)
    boolean_flag(parser, 'nstep-loss-off', default=False)
    parser.add_argument('--saving-folder', type=str, default='/home/giulia/tmp/prova')
    boolean_flag(parser, 'execution', default=False)
    parser.add_argument('--model-name', type=str, default='/tmp/models/ddpg-env-Hopper-v1.ckpt-0')
    parser.add_argument('--nb-min-demo', type=int, default=50)
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args

if __name__ == "__main__":
    args = parse_args()
    success_perc(**args)

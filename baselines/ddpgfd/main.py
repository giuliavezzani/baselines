import argparse
import time
import os
import pickle
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
import tensorflow as tf
from mpi4py import MPI

class Demo():
    def __init__(self, obs0, obs1,obsn, acts, rewards, terms, termsn, rewardsn):
        self.obs0 = obs0
        self.obs1 = obs1
        self.obsn = obsn
        self.acts = acts
        self.rewards = rewards
        self.terms = terms
        self.rewardsn = rewardsn
        self.termsn = termsn

def run(env_id, seed, noise_type, layer_norm, evaluation, demo_file,  alpha, eps, eps_d, target_period_update, lambda_3, nb_training_bc,t_inner_steps,n_value,gamma,lambda_n, behaviour_cloning_off, priorization_off, nstep_loss_off, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create envs.
    env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    gym.logger.setLevel(logging.WARN)

    if evaluation and rank==0:
        eval_env = gym.make(env_id)
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        env = bench.Monitor(env, None)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.

    # Read the doemonstration
    demonstrations = read_demo_file(demo_file, n_value, gamma)
    nb_min_demo = len(demonstrations.obs0)

    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape, nb_min_demo=nb_min_demo, demonstrations=demonstrations, eps_d=eps_d)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=env, eval_env=eval_env, param_noise=param_noise, n_value= n_value,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, gamma=gamma, behaviour_cloning_off=behaviour_cloning_off, priorization_off=priorization_off, nstep_loss_off=nstep_loss_off,
        eps=eps, eps_d=eps_d, lambda_3=lambda_3,lambda_n=lambda_n, alpha=alpha, target_period_update=target_period_update, nb_training_bc=nb_training_bc,t_inner_steps=t_inner_steps, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))

def read_demo_file(demo_file, n_value, gamma):

    demo_dict = np.load(demo_file)
    obs0 = list()
    obs1 = list()
    obsn = list()
    acts = list()
    rewards = list()
    terms = list()
    rewardsn = list()
    termsn = list()

    for i in range(0,len(demo_dict) - 1):
        obs0.append( demo_dict[i]['s0'] )
        obs1.append( demo_dict[i]['s1'] )
        acts.append( demo_dict[i]['a'] )
        rewards.append( demo_dict[i]['r'] )
        terms.append( demo_dict[i]['t'] )
        obsn_single = np.zeros(shape=demo_dict[i]['s0'].shape)
        termn_single = np.zeros(shape=demo_dict[i]['t'].shape)
        rewn_single = np.zeros(shape=demo_dict[i]['r'].shape)
        for j in range(0,len(demo_dict[i]['s0'])):
            if (j + n_value <=  len(demo_dict[i]['s0']) - 1):
                obsn_single[j] = demo_dict[i]['s0'][j + n_value]
                termn_single[j] = demo_dict[i]['t'][j + n_value].astype('float32')
                rewn= 0.
                for t in range(j, j + n_value):
                     rewn +=  gamma ** (t - j) * demo_dict[i]['r'][t]
            else:
                obsn_single[j] = demo_dict[i]['s0'][len(demo_dict[i]['s0']) - 1]
                termn_single[len(demo_dict[i]['s0']) - 1]= 1.0
                rewn= 0.
                termn_single[j]= 1.0
                for t in range(j, len(demo_dict[i]['s0'])):
                     rewn +=  gamma ** (t - j) * demo_dict[i]['r'][t]

            rewn_single[j] = rewn


        termsn.append(termn_single.astype('bool'))
        obsn.append(obsn_single)
        rewardsn.append(rewn_single)


    #import IPython
    #IPython.embed()
    print('obs0', obs0[0].shape)
    print('obs1', obs1[0].shape)
    print('obsn', obsn[0].shape)
    print('acts', acts[0].shape)
    print('reward', rewards[0].shape)
    print('terms', terms[0].shape)
    print('rewardsn', rewardsn[0].shape)
    print('termsn', termsn[0].shape)

    return Demo(np.vstack(obs0), np.vstack(obs1),np.vstack(obsn), np.vstack(acts), np.concatenate(rewards), np.concatenate(terms), np.concatenate(termsn), np.concatenate(rewardsn))

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='Hopper-v1')
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
    parser.add_argument('--nb-training-bc', type=int, default=200) # number of behaviour_cloning training step to be performed
    parser.add_argument('--t-inner-steps', type=int, default=20)
    parser.add_argument('--n-value', type=int, default=5)
    boolean_flag(parser, 'evaluation', default=False)
    parser.add_argument('--lambda-n', type=float, default=0.1) # weight for priorization computation
    boolean_flag(parser, 'behaviour-cloning-off', default=False)
    boolean_flag(parser, 'priorization-off', default=False)
    boolean_flag(parser, 'nstep-loss-off', default=False)
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    # Run actual script.
    run(**args)

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
    def __init__(self, obs0, obs1, acts, rewards, terms):
        self.obs0 = obs0
        self.obs1 = obs1
        self.acts = acts
        self.rewards = rewards
        self.terms = terms

def run(env_id, seed, noise_type, layer_norm, evaluation, demo_file, nb_min_demo, alpha, eps, eps_d, target_period_update, lambda_3, nb_training_bc, **kwargs):
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
    demonstrations = read_demo_file(demo_file)
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape, nb_min_demo=nb_min_demo, demonstrations=demonstrations, alpha=alpha)
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
    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory,
        eps=eps, eps_d=eps_d, lambda_3=lambda_3, target_period_update=target_period_update, nb_training_bc=nb_training_bc, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))

def read_demo_file(demo_file):

    demo_dict = np.load(demo_file)
    obs0 = list()
    obs1 = list()
    acts = list()
    rewards = list()
    terms = list()

    for i in range(len(demo_dict)):
        obs0.append( demo_dict[i]['s0'] )
        obs1.append( demo_dict[i]['s1'] )
        acts.append( demo_dict[i]['a'] )
        rewards.append( demo_dict[i]['r'] )
        terms.append( demo_dict[i]['t'] )

    print('obs0', obs0[0].shape)
    print('obs1', obs1[0].shape)
    print('acts', acts[0].shape)
    print('reward', rewards[0].shape)
    print('terms', terms[0].shape)

    return Demo(obs0, obs1, acts, rewards, terms)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='HalfCheetah-v1')
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
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    parser.add_argument('--nb-min-demo', type=int, default=10) # minimum number of demo guaranteed in the replay buffer
    parser.add_argument('--demo-file', type=str, default='demo-collected.npy') # minimum number of demo guaranteed in the replay buffer
    parser.add_argument('--alpha', type=float, default=0.3) # alpha value for priorization
    parser.add_argument('--eps', type=float, default=0.3) # constant for priorization computation
    parser.add_argument('--eps_d', type=float, default=0.3) # constant for priorization computation
    parser.add_argument('--lambda-3', type=float, default=0.3) # weight for priorization computation
    parser.add_argument('--target-period-update', type=int, default=20) # target networks are updated every target_period_update training steps
    parser.add_argument('--nb-training-bc', type=int, default=20) # number of behaviour_cloning training step to be performed
    boolean_flag(parser, 'evaluation', default=False)
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

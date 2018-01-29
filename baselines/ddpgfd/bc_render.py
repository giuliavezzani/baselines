import os
import time
from collections import deque
import pickle

from baselines.ddpgfd.ddpgfd import DDPGFD
from baselines.ddpgfd.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI


def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise, saving_folder,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, batch_size_bc,priorization_off, nstep_loss_off,
    memory, eps, eps_d, lambda_3, lambda_n, alpha,  target_period_update, nb_training_bc,t_inner_steps,n_value, behaviour_cloning_off,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPGFD(actor, critic, memory, env.observation_space.shape, env.action_space.shape, eps, eps_d,
        lambda_3, priorization_off=priorization_off,nstep_loss_off=nstep_loss_off, lambda_n=lambda_n, alpha=alpha, n_value=n_value,gamma=gamma, tau=tau, normalize_returns=normalize_returns,
        normalize_observations=normalize_observations,batch_size=batch_size, batch_size_bc=batch_size_bc,
        t_inner_steps=t_inner_steps, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm, reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)

    loss_values = np.zeros(nb_training_bc)
    steps = np.arange(0,nb_training_bc )
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        if not behaviour_cloning_off:
            for t_train_bc in range(nb_training_bc):
                #print('---- bc step -----', t_train_bc)
                agent.behaviour_cloning()

                loss_values[t_train_bc] = agent.loss

        agent.reset()
        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False




        #for epoch in range(nb_epochs):
        #    for cycle in range(nb_epoch_cycles):
        #        # Collect more rollouts
        #        rollout_acts = []
        #        rollout_obs0 = []
        #        rollout_obs1 = []
        #        rollout_rews = []
        #        rollout_terms1 = []
        for t_rollout in range(nb_rollout_steps):
            # Predict next action.
            print('num roll', t_rollout)
            start_time = time.time()

            #print('num rollout collected: ', t_rollout)

            if done:
                obs = env.reset()
                agent.reset()
                done = False
            while not done:
                env.render()
                action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                assert action.shape == env.action_space.shape

                # Execute next action.
                if rank == 0 and render:
                    env.render()
                assert max_action.shape == action.shape

                new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPGfD is concerned, every action is in [-1, 1])

                if rank == 0 and render and epoch/nb_epochs >= 0.8:
                    env.render()
                # Book-keeping.

                obs = new_obs

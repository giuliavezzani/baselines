import gym
import pickle
import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U

def main():
    env = gym.make('Hopper-v1')
    f = open('model.pkl', 'rb')
    act = pickle.load(f)

    returns = []
    observations0 = []
    actions = []
    observations1 = []
    terminals = []
    experts = []

    #while True:
    for i_rollout in range(10):
        print('rollout_no: ', i_rollout)
        obs = env.reset()
        episode_rew = 0
        done = False

        while not done:
            env.render()
            #action = act(obs[None, :])
            observations0.append(obs)
            action = env.action_space.sample()
            obs, rew, done, _ = env.step(action)
            actions.append(action)
            observations1.append(obs)
            episode_rew += rew
            terminals.append(done)
            returns.append(episode_rew)

        expert_data = {'s0': np.array(observations0),
                       's1': np.array(observations1),
                       'a': np.array(actions),
                       'r': np.array(returns),
                       't': np.array(terminals)}
        print(np.array(observations0).shape)
        print(np.array(observations1).shape)
        print(np.array(actions).shape)
        print(np.array(returns).shape)
        print(np.array(terminals).shape)
    experts.append(expert_data)

    np.save('demo-collected-2.npy', experts)

if __name__ == '__main__':
    main()

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
    for i_rollout in range(100):
        print('rollout_no: ', i)
        obs = env.reset()
        observations0.append(obs)
        print(obs)
        episode_rew = 0
        done = False
        while not done:
            env.render()
            #action = act(obs[None, :])
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
       experts.append(expert_data)

   np.save('demo-collected.npy', experts)

if __name__ == '__main__':
    main()

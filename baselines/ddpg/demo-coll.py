import gym
import pickle
import tensorflow as tf
import baselines.common.tf_util as U

def main():
    env = gym.make('Hopper-v1')
    f = open('model.pkl', 'rb')
    act = pickle.load(f)

    #while True:
    for i_episode in range(100):
        obs = env.reset()
        print(obs)
        episode_rew = 0
        done = False
        while not done:
            env.render()

            obs, rew, done, _ = env.step(act(obs[None, :]))
            episode_rew += rew

if __name__ == '__main__':
    main()

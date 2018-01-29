import matplotlib.pyplot as plt
from baselines import logger
import pandas
import argparse
import numpy as np
from matplotlib.pyplot import cm




def plot(file_path_ddpg, file_path_ddpgfd,label1, label2, **kwargs):
    data_ddpg = logger.read_csv(file_path_ddpg)
    data_ddpgfd = logger.read_csv(file_path_ddpgfd)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    count = 1
    color=cm.rainbow(np.linspace(0,1,26))

    for key in sorted(data_ddpg.keys()):
        x_ddpg = np.arange(0,len(data_ddpg[key]))
        y_ddpg = data_ddpg[key]
        c = color[(count - 1)%26]

        plt.subplot(6,5, count)
        plt.xlabel('Epochs ', fontsize=8)
        plt.ylabel(key, fontsize=8)
        line_ddpg = plt.plot(x_ddpg,y_ddpg, linewidth=2, c=c, label=label1)
        count += 1
        x_ddpgfd = np.arange(0,len(data_ddpgfd[key]))
        y_ddpgfd = data_ddpgfd[key]
        c = color[(count + 1) % 26]
        line_ddpgfd = plt.plot(x_ddpgfd,y_ddpgfd, linewidth=2, c=c, label=label2)
        plt.legend((label1, label2))


    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file-path-ddpg', type=str, default='progress.csv')
    parser.add_argument('--file-path-ddpgfd', type=str, default='progress.csv')
    parser.add_argument('--label1', type=str, default='ddpg')
    parser.add_argument('--label2', type=str, default='ddpgfd')
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameter
    dict_args = vars(args)
    return dict_args

if __name__ == "__main__":
    args = parse_args()
    plot(**args)

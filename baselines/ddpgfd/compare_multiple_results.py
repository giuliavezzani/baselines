import matplotlib.pyplot as plt
from baselines import logger
import pandas
import argparse
import numpy as np
from matplotlib.pyplot import cm




def plot(file_path_ddpg,  file_path_ddpgfd1, file_path_ddpgfd2, file_path_ddpgfd3, file_path_ddpgfd4,label1, label2, label3, label4, label5,**kwargs):
    data_ddpg = logger.read_csv(file_path_ddpg)
    data_ddpgfd1 = logger.read_csv(file_path_ddpgfd1)
    data_ddpgfd2 = logger.read_csv(file_path_ddpgfd2)
    data_ddpgfd3 = logger.read_csv(file_path_ddpgfd3)
    data_ddpgfd4 = logger.read_csv(file_path_ddpgfd4)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    count = 1
    color=cm.rainbow(np.linspace(0,1,26))

    for key in sorted(data_ddpg.keys()):
        x_ddpg = np.arange(0,len(data_ddpg[key]))
        y_ddpg = data_ddpg[key]
        c = color[count - 1]
        plt.subplot(5,6, count)
        plt.xlabel('Epochs ', fontsize=8)
        plt.ylabel(key, fontsize=8)
        line_ddpg = plt.plot(x_ddpg,y_ddpg, linewidth=2, c=c, label=label1)
        count += 1
        x_ddpgfd1 = np.arange(0,len(data_ddpgfd1[key]))
        y_ddpgfd1 = data_ddpgfd1[key]
        c = color[(count + 2) % 26]
        line_ddpgfd1 = plt.plot(x_ddpgfd1,y_ddpgfd1, linewidth=2, c=c, label=label2)

        x_ddpgfd2 = np.arange(0,len(data_ddpgfd2[key]))
        y_ddpgfd2 = data_ddpgfd2[key]
        c = color[(count + 6) % 26]
        line_ddpgfd2 = plt.plot(x_ddpgfd2,y_ddpgfd2, linewidth=2, c=c, label=label3)

        x_ddpgfd3 = np.arange(0,len(data_ddpgfd3[key]))
        y_ddpgfd3 = data_ddpgfd3[key]
        c = color[(count +10) % 26]
        line_ddpgfd = plt.plot(x_ddpgfd3,y_ddpgfd3, linewidth=2, c=c, label=label4)

        x_ddpgfd4 = np.arange(0,len(data_ddpgfd4[key]))
        y_ddpgfd4 = data_ddpgfd4[key]
        c = color[(count + 14) % 26]
        line_ddpgfd4 = plt.plot(x_ddpgfd4,y_ddpgfd4, linewidth=2, c=c, label=label5)
        plt.legend((label1, label2, label3, label4, label5), prop={'size': 4})


    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file-path-ddpg', type=str, default='progress.csv')
    parser.add_argument('--file-path-ddpgfd1', type=str, default='progress.csv')
    parser.add_argument('--file-path-ddpgfd2', type=str, default='progress.csv')
    parser.add_argument('--file-path-ddpgfd3', type=str, default='progress.csv')
    parser.add_argument('--file-path-ddpgfd4', type=str, default='progress.csv')
    parser.add_argument('--label1', type=str, default='ddpg')
    parser.add_argument('--label2', type=str, default='ddpg+demo')
    parser.add_argument('--label3', type=str, default='ddpg+bc+prior')
    parser.add_argument('--label4', type=str, default='ddpg+nstep-loss')
    parser.add_argument('--label5', type=str, default='ddpg+multiple-learning')

    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameter
    dict_args = vars(args)
    return dict_args

if __name__ == "__main__":
    args = parse_args()
    plot(**args)

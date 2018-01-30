import matplotlib.pyplot as plt
from baselines import logger
import pandas
import argparse
import numpy as np
from matplotlib.pyplot import cm




def plot(file_path1, file_path2, file_path3, file_path4, file_path5,label1,  **kwargs):
    data_ddpg1 = logger.read_csv(file_path1)
    data_ddpg2 = logger.read_csv(file_path2)
    data_ddpg3 = logger.read_csv(file_path3)
    data_ddpg4 = logger.read_csv(file_path4)
    data_ddpg5 = logger.read_csv(file_path5)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    count = 1
    color=cm.rainbow(np.linspace(0,1,26))

    count = 5
    y_ddpg1 = np.arange(0,len(data_ddpg1['AverageReturn']) +1)
    for key in sorted(data_ddpg1.keys()):
        if key=='AverageReturn':
            x_ddpg1 = np.arange(0,len(data_ddpg1[key]) +1)
            y_ddpg1 = data_ddpg1[key]

    y_ddpg2 = np.arange(0,len(data_ddpg2['AverageReturn']) +1)
    for key in sorted(data_ddpg2.keys()):
        if key=='AverageReturn':
            x_ddpg2 = np.arange(0,len(data_ddpg2[key]))
            y_ddpg2 = data_ddpg2[key]

    y_ddpg3 = np.arange(0,len(data_ddpg3['AverageReturn']) +1)
    for key in sorted(data_ddpg3.keys()):
        if key=='AverageReturn':
            x_ddpg3 = np.arange(0,len(data_ddpg3[key]))
            y_ddpg3 = data_ddpg3[key]

    y_ddpg4 = np.arange(0,len(data_ddpg4['AverageReturn']) +1)
    for key in sorted(data_ddpg4.keys()):
        if key=='AverageReturn':
            x_ddpg4 = np.arange(0,len(data_ddpg4[key]))
            y_ddpg4 = data_ddpg4[key]

    y_ddpg5 = np.arange(0,len(data_ddpg5['AverageReturn']) +1)
    for key in sorted(data_ddpg5.keys()):
        if key=='AverageReturn':
            x_ddpg5 = np.arange(0,len(data_ddpg5[key]))
            y_ddpg5 = data_ddpg5[key]


    y_mean = (y_ddpg1 + y_ddpg2 +  y_ddpg3 + y_ddpg4 + y_ddpg5 ) / 5.0

    print(y_mean)

    y_std = np.arange(0,len(data_ddpg4['AverageReturn']))

    y_std = (y_mean - y_ddpg1) ** 2 +  (y_mean - y_ddpg2) ** 2 +  (y_mean - y_ddpg3) ** 2 +  (y_mean - y_ddpg4) ** 2 +  (y_mean - y_ddpg5) ** 2

    y_std = np.sqrt(y_std / (count - 1) )
    c = color[(count - 1)%26]

    fig, ax = plt.subplots(1)
    plt.xlabel('Epochs ', fontsize=8)
    plt.ylabel(key, fontsize=8)
    line_ddpg = plt.plot(x_ddpg1 ,y_mean, linewidth=2, c=c, label=label1)
    ax.fill_between(x_ddpg1, y_mean + y_std, y_mean -  y_std, facecolor=c, alpha=0.5)



    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file-path1', type=str, default='0/progress.csv')
    parser.add_argument('--file-path2', type=str, default='1/progress.csv')
    parser.add_argument('--file-path3', type=str, default='2/progress.csv')
    parser.add_argument('--file-path4', type=str, default='3/progress.csv')
    parser.add_argument('--file-path5', type=str, default='4/progress.csv')
    parser.add_argument('--label1', type=str, default='DDPGHER')
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameter
    dict_args = vars(args)
    return dict_args

if __name__ == "__main__":
    args = parse_args()
    plot(**args)

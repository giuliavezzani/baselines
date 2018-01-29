import matplotlib.pyplot as plt
from baselines import logger
import pandas
import argparse
import numpy as np
from matplotlib.pyplot import cm




def plot(file_path_ddpg, file_path_ddpgfd1, file_path_ddpgfd2, file_path_ddpgfd3, file_path_ddpgfd4, file_path_ddpgfd5, file_path_ddpgfd6, file_path_ddpgfd7,  label1, label2, label3, label4, label5,label6, label7, label8,**kwargs):
    print('DEBUG', file_path_ddpg)
    data_ddpg = logger.read_csv(file_path_ddpg)
    data_ddpgfd1 = logger.read_csv(file_path_ddpgfd1)
    data_ddpgfd2 = logger.read_csv(file_path_ddpgfd2)
    data_ddpgfd3 = logger.read_csv(file_path_ddpgfd3)
    data_ddpgfd4 = logger.read_csv(file_path_ddpgfd4)
    data_ddpgfd5 = logger.read_csv(file_path_ddpgfd5)
    data_ddpgfd6 = logger.read_csv(file_path_ddpgfd6)
    data_ddpgfd7 = logger.read_csv(file_path_ddpgfd7)

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


        x_ddpgfd6 = np.arange(0,len(data_ddpgfd5[key]))
        y_ddpgfd6 = data_ddpgfd5[key]
        c = color[(count + 18) % 26]
        line_ddpgfd6 = plt.plot(x_ddpgfd6,y_ddpgfd6, linewidth=2, c=c, label=label6)

        x_ddpgfd7 = np.arange(0,len(data_ddpgfd6[key]))
        y_ddpgfd7 = data_ddpgfd6[key]
        c = color[(count +22) % 26]
        line_ddpgfd7 = plt.plot(x_ddpgfd7,y_ddpgfd7, linewidth=2, c=c, label=label7)

        x_ddpgfd8 = np.arange(0,len(data_ddpgfd7[key]))
        y_ddpgfd8 = data_ddpgfd7[key]
        c = color[(count + 24) % 26]
        line_ddpgfd8= plt.plot(x_ddpgfd8,y_ddpgfd8, linewidth=2, c=c, label=label8)
        plt.legend((label1, label2, label3, label4, label5, label6, label7, label8), prop={'size': 4})


    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file-path-ddpg', type=str, default='1/progress.csv')
    parser.add_argument('--file-path-ddpgfd1', type=str, default='2/progress.csv')
    parser.add_argument('--file-path-ddpgfd2', type=str, default='3/progress.csv')
    parser.add_argument('--file-path-ddpgfd3', type=str, default='4/progress.csv')
    parser.add_argument('--file-path-ddpgfd4', type=str, default='5/progress.csv')
    parser.add_argument('--file-path-ddpgfd5', type=str, default='6/progress.csv')
    parser.add_argument('--file-path-ddpgfd6', type=str, default='7/progress.csv')
    parser.add_argument('--file-path-ddpgfd7', type=str, default='8/progress.csv')
    parser.add_argument('--label1', type=str, default='ddpg-pickeup-1')
    parser.add_argument('--label2', type=str, default='ddpg-pickeup-2')
    parser.add_argument('--label3', type=str, default='ddpg-pickeup-3')
    parser.add_argument('--label4', type=str, default='ddpg-pickeup-4')
    parser.add_argument('--label5', type=str, default='ddpg-pickeup-5')
    parser.add_argument('--label6', type=str, default='ddpg-pickeup-6')
    parser.add_argument('--label7', type=str, default='ddpg-pickeup-7')
    parser.add_argument('--label8', type=str, default='ddpg-pickeup-8')


    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameter
    dict_args = vars(args)
    return dict_args

if __name__ == "__main__":
    args = parse_args()
    plot(**args)

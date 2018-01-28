import matplotlib.pyplot as plt
from baselines import logger
import pandas
import argparse
import numpy as np
from matplotlib.pyplot import cm




def plot(file_path, **kwargs):
    data = logger.read_csv(file_path)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    count = 1
    color=cm.rainbow(np.linspace(0,1,26))

    print(data.keys())

    print(len(data))
    for key in sorted(data.keys()):
        if key == 'AverageReturn':
            x = np.arange(0,len(data[key]))
            y = data[key]
            c = color[(count - 1)%26]
            plt.subplot(1,1, count)
            plt.xlabel('Epochs ', fontsize=8)
            plt.ylabel(key, fontsize=8)
            plt.plot(x,y, linewidth=2, c=c)
            count += 1

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file-path', type=str, default='progress.csv')
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameter
    dict_args = vars(args)
    return dict_args

if __name__ == "__main__":
    args = parse_args()
    plot(**args)

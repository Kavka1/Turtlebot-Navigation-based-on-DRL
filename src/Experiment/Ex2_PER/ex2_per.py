import os
import sys
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)

from train import *
from parser import add_argument
import numpy as np

def main():
    args = add_argument()
    seeds = [30, 50]
    algs = ['DDPG', 'TD3']
    data_paths = {False: 'results/Normal', True: 'results/PER'}

    for i in range(3):
        if i == 0:
            args.seed = 30
            args.alg = 'TD3'
            for flag in [False, True]:
                args.PER = flag
                train_td3(args, data_path=data_paths[flag], save_model=False)
        elif i == 1:
            args.seed = 50
            args.alg = 'DDPG'
            for flag in [False, True]:
                args.PER = flag
                train_ddpg(args, data_path=data_paths[flag], save_model=False)
        else:
            args.seed = 50
            args.alg = 'TD3'
            for flag in [False, True]:
                args.PER = flag
                train_td3(args, data_path=data_paths[flag], save_model=False)

if __name__ == '__main__':
    main()
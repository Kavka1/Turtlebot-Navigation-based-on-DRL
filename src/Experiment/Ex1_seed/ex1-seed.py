import os
import sys
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)

from parser import add_argument
from train import *
from evaluate import Evaluate
import numpy as np

def main():
    args = add_argument()

    algs = [ 'PPO', 'TD3']
    seeds = [10, 20, 30, 40, 50]

    for alg in algs:
        args.alg = alg
        for seed in seeds:
            args.seed = seed
            if alg == 'DDPG':
                train_ddpg(args, save_model=False)
            elif alg == 'PPO':
                train_ppo(args, save_model=False)
            else:
                train_td3(args, save_model=False)



if __name__ == '__main__':
    main()
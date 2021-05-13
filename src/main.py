#!/usr/bin/env python3
import argparse
import torch
from train import train_ddpg, train_td3, train_ppo
from evaluate import Evaluate
from parser import *



if __name__ == '__main__':
    args = add_argument()

    if args.is_training:
        if args.alg == 'TD3':
            train_td3(args)
        elif args.alg == 'DDPG':
            train_ddpg(args)
        elif args.alg == 'PPO':
            train_ppo(args)
        else:
            raise ValueError('Algorithm not chosen or not right')
    else:
        Evaluate(args)
import torch
import rospy
import numpy as np
import time, datetime
from torch.utils.tensorboard import SummaryWriter
from environment import Env
from DDPG import DDPGAgent
from PPO import PPOAgent
from TD3 import TD3Agent


def Evaluate(args):
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    rospy.init_node('navigation')
    env = Env(args)

    if args.alg == 'TD3':
        agent = TD3Agent(s_dim=14, a_dim=2, args=args)
        agent.load_model(
            pi_path='pretrained_models_TD3/policy_{}'.format(args.load_model_remark),
            q1_path='pretrained_models_TD3/q1_{}'.format(args.load_model_remark),
            q2_path='pretrained_models_TD3/q2_{}'.format(args.load_model_remark)
        )
    elif args.alg == 'DDPG':
        agent = DDPGAgent(s_dim=14, a_dim=2, args=args)
        agent.load_model(
            pi_path = 'pretrained_models_DDPG/policy_{}'.format(args.load_model_remark),
            q_path = 'pretrained_models_DDPG/q_{}'.format(args.load_model_remark)
        )
    elif args.alg == 'PPO':
        agent = PPOAgent(s_dim=14, a_dim=2, args=args)
        agent.load_model(
            pi_path='pretrained_models_PPO/policy_{}'.format(args.load_model_remark),
            v_path='pretrained_models_PPO/v_{}'.format(args.load_model_remark)
        )
    else:
        raise NotImplemented

    succeed_count = 0
    total_count = 0
    for i_episode in range(100):
        done = 0
        round_step = 0
        round_reward = 0
        s = env.reset()
        while not done:
            a = agent.choose_action(s)
            s_, r, die, arrive = env.step(a, round_step)
            round_reward += r
            round_step += 1
            s = s_
            if die or arrive or round_step >= 500:
                done = 1
        total_count += 1
        succeed_count += 1 if arrive else 0
        print('Alg: {} | episode: {} | round step: {} | round reward: {} | {} | succeed rate: {:.2f}'.format(
            args.alg, i_episode, round_step, round_reward, 'Success' if arrive else 'Fail', succeed_count / total_count
            )
        )

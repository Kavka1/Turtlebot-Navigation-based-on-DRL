import torch
import rospy
import numpy as np
import time, datetime
from torch.utils.tensorboard import SummaryWriter
from environment import Env
from memory import Memory, PER_Memory
from DDPG import DDPGAgent
from PPO import PPOAgent
from TD3 import TD3Agent


def train_ddpg(args, data_path = 'results', save_model = True):
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    rospy.init_node('navigation')
    env = Env(args)
    agent = DDPGAgent(s_dim=14, a_dim=2, args=args)

    if args.PER:
        memory = PER_Memory(memory_size=args.memory_size, s_dim=14, a_dim=2)
    else:
        memory = Memory(memory_size=args.memory_size, s_dim=14, a_dim=2)

    writer = SummaryWriter('{}/DDPG_seed_{}_{}'.format(data_path, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    start_time = time.time()
    accumulation_r = 0
    total_step = 0
    episode = 0
    for i_episode in range(args.max_episode):
        s = env.reset()
        round_step = 0
        round_reward = 0
        done = 0
        succeed = 0
        while not done:
            a = agent.choose_action(s)
            a[0] = np.clip(np.random.normal(a[0], args.noise_std_ddpg), 0., 1.)
            a[1] = np.clip(np.random.normal(a[1], args.noise_std_ddpg), -1., 1.)
            s_, r, die, arrive = env.step(a, round_step)

            if args.PER:
                p = agent.compute_priority(s, a, r, s_)
                memory.store(s, a, r, s_, die or arrive, p)
            else:
                memory.store(s, a, r, s_, die or arrive)

            s = s_

            accumulation_r += r
            round_reward += r
            total_step += 1
            round_step += 1

            if total_step > args.training_start:
                q_loss, pi_loss = agent.learn(memory)
                writer.add_scalar('Loss/q_loss', q_loss, total_step)
                writer.add_scalar('Loss/pi_loss', pi_loss, total_step)

            if total_step % 10 == 0 and total_step > args.exploration_decay_start:
                args.noise_std_ddpg *= 0.9999

            if round_step>=500 or die or arrive:
                done = 1
                succeed = 1 if arrive else 0

        print(
            'training time: {:.2f} | episode: {} | round step: {} | total steps: {} | round reward: {} | {}'.format(
                (time.time() - start_time) / 60., episode, round_step, total_step, round_reward, 'Succeed' if succeed else 'Failed'
            )
        )
        if episode % args.log_interval == 0:
            avg_r = accumulation_r / args.log_interval
            writer.add_scalar('average reward per {} episode'.format(args.log_interval), avg_r, total_step)
            accumulation_r = 0
        if episode % 250 == 0 and save_model:
            agent.save_model(remarks='episode_{}_step_{}'.format(episode, total_step))

        episode += 1


def train_td3(args, data_path = 'results', save_model = True):
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    rospy.init_node('navigation')
    env = Env(args)
    agent = TD3Agent(s_dim=14, a_dim=2, args=args)

    if args.PER:
        memory = PER_Memory(memory_size=args.memory_size, s_dim=14, a_dim=2)
    else:
        memory = Memory(memory_size=args.memory_size, s_dim=14, a_dim=2)

    writer = SummaryWriter('{}/TD3_seed_{}_{}'.format(data_path, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    start_time = time.time()
    accumulation_r = 0
    total_step = 0
    episode = 0
    for i_episode in range(args.max_episode):
        s = env.reset()
        done = 0
        succeed = 0
        round_step = 0
        round_reward = 0
        while not done:
            a = (agent.choose_action(s) + np.random.normal(0., args.explore_noise_std, size=2)).clip(-1., 1.)
            s_, r, die, arrive = env.step(a, round_step)

            if args.PER:
                p = agent.compute_priority(s, a, r, s_)
                memory.store(s, a, r, s_, die or arrive, p)
            else:
                memory.store(s, a, r, s_, die or arrive)

            if total_step >= args.training_start:
                q1_loss, q2_loss, pi_loss = agent.learn(memory, total_step)

                writer.add_scalar('Loss/q1_loss', q1_loss, total_step)
                writer.add_scalar('Loss/q2_loss', q2_loss, total_step)
                writer.add_scalar('Loss/pi_loss', pi_loss, total_step)

            accumulation_r += r
            round_reward += r
            total_step += 1
            s = s_
            round_step += 1

            if round_step >= 500 or die or arrive:
                done = 1
                succeed = 1 if arrive else 0

        print(
            'training time: {:.2f} | episode: {} | round step: {} | total steps: {} | round reward: {} | {}'.format(
                (time.time() - start_time) / 60., episode, round_step, total_step, round_reward, 'Succeed' if succeed else 'Failed'
            )
        )
        if episode % args.log_interval == 0:
            avg_r = accumulation_r / args.log_interval
            writer.add_scalar('average reward per {} episode'.format(args.log_interval), avg_r, total_step)
            accumulation_r = 0
        if episode % 250 == 0 and save_model:
            agent.save_model(remarks='episode_{}_step_{}'.format(episode, total_step))

        episode += 1


def train_ppo(args, data_path = 'results', save_model = True):
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    rospy.init_node('navigation')
    env = Env(args)
    agent = PPOAgent(s_dim=14, a_dim=2, args=args)

    writer = SummaryWriter('{}/PPO_seed_{}_{}'.format(data_path, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    start_time = time.time()
    accumulation_r = 0
    total_step = 0
    episode = 0
    for i_episode in range(args.max_episode):
        s = env.reset()
        done = 0
        round_step = 0
        round_reward = 0
        succeed = 0
        while not done:
            a = agent.choose_action(s)
            s_, r, die, arrive = env.step(a, round_step)

            agent.memory.rewards.append(r)
            agent.memory.is_terminals.append(die or arrive)

            if total_step % args.update_timestep == 0 and total_step > 0:
                agent.learn()
                agent.memory.clear()

            s = s_
            accumulation_r += r
            round_reward += r
            total_step += 1
            round_step += 1

            if round_step >= 500 or die or arrive:
                done = 1
                succeed = 1 if arrive else 0

        print(
            'training time: {:.2f} | episode: {} | round step: {} | total steps: {} | round reward: {} | {}'.format(
                (time.time() - start_time) / 60., episode, round_step, total_step, round_reward, 'Succeed' if succeed else 'Failed'
            )
        )
        if episode % args.log_interval == 0:
            avg_r = accumulation_r / args.log_interval
            writer.add_scalar('average reward per {} episode'.format(args.log_interval), avg_r, total_step)
            accumulation_r = 0
        if episode % 250 == 0 and save_model:
            agent.save_model(remarks='episode_{}_step_{}'.format(episode, total_step))

        episode += 1
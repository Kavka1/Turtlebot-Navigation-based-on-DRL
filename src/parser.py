import argparse

def add_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--alg', type=str, default='PPO')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--load_model_remark', type=str, default='episode_6000_step_179833')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--lr_pi', type=float, default=3e-4)
    parser.add_argument('--lr_q', type=float, default=3e-4)
    parser.add_argument('--memory_size', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--training_start', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--max_episode', type=int, default=650)
    #whether priority experience replay
    parser.add_argument('--PER', type= bool, default=False)
    #td3 hyperparameters
    parser.add_argument('--noise_std_td3', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--explore_noise_std', type=float, default=0.1)
    parser.add_argument('--policy_update_interval', type=int, default=2)
    #DDPG hyperparameters
    parser.add_argument('--weight_decay_q', type=float, default=0.01)
    parser.add_argument('--exploration_decay_start', type=int, default=50000)
    parser.add_argument('--noise_std_ddpg', type=float, default=1.)
    #PPO hyperparameters
    parser.add_argument('--update_timestep', type=int, default=250)
    parser.add_argument('--lamda', type=float, default=0.95)
    parser.add_argument('--lr_ppo', type=float, default=3e-4)
    parser.add_argument('--action_std', type=float, default=0.5)
    parser.add_argument('--K_epoch', type=int, default=50)
    parser.add_argument('--epsilon_clip', type=float, default=0.2)

    args = parser.parse_args()
    return args


import argparse as AP
import os
import torch
import numpy as np
import ddpg as DDPG
import env as ENV


def set_args():
    cfg = AP.ArgumentParser(description='args for env, agent and essential info')
    
    # 环境参数
    cfg.add_argument('--f_w', default = 1000, type = float, \
        help = 'the width of the charging field')
    cfg.add_argument('--f_h', default = 1000, type = float, \
        help = 'the height of the charging field')
    cfg.add_argument('--s_N', default = 80, type = int, \
        help = 'the num of sensors')
    cfg.add_argument('--b_C', default = 2, type = float, \
        help = 'the battery capacity of each sensor')
    cfg.add_argument('--p_c', default = 0.9, type = float, \
        help = 'the charging power of the mobile charger')
    cfg.add_argument('--p_th', default = 0.001, type = float, \
        help = 'the charging threshold of the sensor')
    cfg.add_argument('--alpha', default = 36, type = float, \
        help = 'alpha in wireless charging model')
    cfg.add_argument('--beta', default = 30, type = float, \
        help = 'beta in wireless charging model')
    cfg.add_argument('--m_p', default = 5.6, type = float, \
        help = 'the price for a unit move')
    
    # 智能体参数
    cfg.add_argument('--ep', default = 300, type = int, \
        help = 'training episodes')
    cfg.add_argument('--memory_cap', default = 512, type = int, \
        help = 'capacity of memory pool')
    cfg.add_argument('--batch_size', default = 16, type = int, \
        help = 'batch size for sampling')
    cfg.add_argument('--gamma', default = 0.8, type = float, \
        help = 'the discount factor')
    cfg.add_argument('--eta_actor', default = 0.001, type = float, \
        help = 'the learning rate for actor')
    cfg.add_argument('--eta_critic', default = 0.001, type = float, \
        help = 'the learning rate for critic')
    cfg.add_argument('--tau', default = 0.01, type = float, \
        help = 'the update rate')
    cfg.add_argument('--hidden_dim', default = 128, type = int, \
        help = 'num of neurals in each hidden layer')
    cfg.add_argument('--hidden_layer', default = 2, type = int, \
        help = 'num of hidden layers')

    # 结果存储路径
    cwd = os.getcwd()
    cfg.add_argument('--model_path', default = cwd + '/model/')
    cfg.add_argument('--reward_path', default = cwd + '/reward/')
    
    args = cfg.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.state_dim = args.s_N + 2
    args.action_dim = 2

    return args


def train(cfg, env, agent):
    ep_rewards = []
    for i in range(1, cfg.ep + 1):
        step = 0
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            step += 1
            action = agent.choose_action(state, noise = True)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            agent.memory_pool.add(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            # print('action:', action)
            # print('reward:', reward)
        ep_rewards.append(ep_reward)
        print('Episode:{}, Cumulative Reward:{}, Total Steps:'\
            .format(i, ep_reward, step))
    agent.save_actor()
    return ep_rewards


if __name__ == '__main__':
    
    # 创建一个存储参数的对象
    cfg = set_args()
    # 创建智能体和环境对象
    env = ENV.ENV(cfg)
    agent = DDPG.DDPG(cfg)
    # 训练
    # train(cfg, env, agent)
    # 绘图

    # 保存参数

    # 加载参数

    # 测试

    # 绘图


    # a=np.random.random([2,cfg.state_dim])*2
    # b= torch.FloatTensor(a)
    # actor = DDPG.Actor(cfg.state_dim, cfg.hidden_dim, cfg.hidden_layer, \
    #     cfg.action_dim)
    # print(actor(b, noise = True))
    # print(actor(b, noise = True))
    # if torch.cuda.is_available():
    #     print('111111111111')
    # else:
    #     print('00000000000000000')

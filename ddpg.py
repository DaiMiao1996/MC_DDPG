import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
import os


class Actor(nn.Module):
    def __init__(self, in_features, hidden_dim, hidden_layer, \
        out_features, bias = True, init_w = 1e-2):
        super(Actor, self).__init__()
        self.bias = bias
        self.init_w = init_w
        # 此处不能用简单列表，否则优化器会报：optimizer got an empty parameter list
        self.linears = nn.ModuleList()
        self.noises = nn.ModuleList()
        # input layer
        self.linears.append(nn.Linear(in_features, hidden_dim, bias))
        self.linears[-1].weight.data.uniform_(-init_w, init_w)
        self.noises.append(nn.Linear(in_features, hidden_dim, bias))
        self.noises[-1].weight.data.normal_(0, init_w / 10)
        if bias:
            self.linears[-1].bias.data.uniform_(-init_w, init_w)
            self.noises[-1].bias.data.normal_(0, init_w / 10)
        for i in range(hidden_layer):
            # hidden layer
            self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias))
            self.linears[-1].weight.data.uniform_(-init_w, init_w)
            self.noises.append(nn.Linear(hidden_dim, hidden_dim, bias))
            self.noises[-1].weight.data.normal_(0, init_w / 10)
            if bias:
                self.linears[-1].bias.data.uniform_(-init_w, init_w)
                self.noises[-1].bias.data.normal_(0, init_w / 10)
        # output layer
        self.linears.append(nn.Linear(hidden_dim, out_features, bias))
        self.linears[-1].weight.data.uniform_(-init_w, init_w)
        self.noises.append(nn.Linear(hidden_dim, out_features, bias))
        self.noises[-1].weight.data.normal_(0, init_w / 10)
        if bias:
            self.linears[-1].bias.data.uniform_(-init_w, init_w)
            self.noises[-1].bias.data.normal_(0, init_w / 10)

    def forward(self, x, noise = False):
        for i in range(len(self.linears) - 1):
            if not noise:
                x = F.relu(self.linears[i](x))
            else:
                if self.bias:
                    self.noises[i].bias.data.normal_(0, self.init_w)
                self.noises[i].weight.data.normal_(0, self.init_w)
                x = F.relu(self.linears[i](x) + self.noises[i](x))
        if not noise:
            x = torch.sigmoid(self.linears[-1](x)) # nn.sigmoid()已不被推荐
        else:
            if self.bias:
                self.noises[-1].bias.data.normal_(0, self.init_w)
            self.noises[-1].weight.data.normal_(0, self.init_w)
            x = torch.sigmoid(self.linears[-1](x) + self.noises[-1](x))
        return x



class Critic(nn.Module):
    def __init__(self, in_features, hidden_dim, hidden_layer, \
        bias = True, init_w=1e-2):
        super(Critic, self).__init__()
        self.linears = nn.ModuleList()
        # input layer
        self.linears.append(nn.Linear(in_features, hidden_dim, bias))
        self.linears[-1].weight.data.uniform_(-init_w, init_w)
        if bias:
            self.linears[-1].bias.data.uniform_(-init_w, init_w)
        for i in range(hidden_layer):
            # hidden layer
            self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias))
            self.linears[-1].weight.data.uniform_(-init_w, init_w)
            if bias:
                self.linears[-1].bias.data.uniform_(-init_w, init_w)
        # output layer
        self.linears.append(nn.Linear(hidden_dim, 1, bias))
        self.linears[-1].weight.data.uniform_(-init_w, init_w)
        if bias:
            self.linears[-1].bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        for i in range(len(self.linears)-1):
            x = F.relu(self.linears[i](x))
        x = self.linears[-1](x)
        return x



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pointer = 0

    def add(self, state, action, reward, next_state, done):
        '''
        substitute the most obsolete element with the newest one
        '''
        if len(self.buffer) == self.capacity:
            self.buffer[self.pointer] = (state, action, reward, next_state, done)
            self.pointer = (self.pointer + 1) % self.capacity
        else:
            self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        '''
        fetch a batch of elements from the memory pool, if possible
        '''
        if self.capacity < batch_size:
            return None
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
    
    def len(self):
        return len(self.buffer)



class DDPG:
    def __init__(self, cfg):
        self.be_actor = Actor(cfg.state_dim, cfg.hidden_dim, \
            cfg.hidden_layer, cfg.action_dim).to(cfg.device)
        self.be_critic = Critic(cfg.state_dim + cfg.action_dim, \
            cfg.hidden_dim, cfg.hidden_layer).to(cfg.device)
        self.tar_actor = Actor(cfg.state_dim, cfg.hidden_dim, \
            cfg.hidden_layer, cfg.action_dim).to(cfg.device)
        self.tar_critic = Critic(cfg.state_dim + cfg.action_dim, \
            cfg.hidden_dim, cfg.hidden_layer).to(cfg.device)
        
        self.batch_size = cfg.batch_size
        self.gamma = cfg.gamma
        
        for tar_param, be_param in zip(self.tar_critic.parameters(), \
            self.be_critic.parameters()):
            tar_param.data.copy_(be_param.data)
        for tar_param, be_param in zip(self.tar_actor.parameters(), \
            self.be_actor.parameters()):
            tar_param.data.copy_(be_param.data)

        self.actor_opt = optim.Adam(self.be_actor.parameters(), lr = cfg.eta_actor)
        self.critic_opt = optim.Adam(self.be_critic.parameters(), lr = cfg.eta_critic)
        self.memory_pool = ReplayBuffer(cfg.memory_cap)
        self.device = cfg.device
        self.tau = cfg.tau
        self.model_path = cfg.model_path
        self.reward_path = cfg.reward_path
    
    def choose_action(self, state, noise = False):
        state = torch.FloatTensor(state).to(self.device)
        action = self.be_actor(state, noise)
        return action.detach().cpu().numpy()
    
    def update(self):
        if self.memory_pool.len() <self.batch_size:
            return
        else:
            states, actions, rewards, next_states, dones = \
                self.memory_pool.sample(self.batch_size)
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.FloatTensor(np.array(actions)).to(self.device)
            rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

            ys = rewards + (1 - dones) * self.gamma * self.tar_critic(torch.cat\
                ((next_states, self.tar_actor(next_states).detach()), dim = 1))
            mse_loss = nn.MSELoss()(ys, self.be_critic(torch.cat\
                ((states, actions), dim = 1)))
            self.critic_opt.zero_grad()
            mse_loss.backward()
            self.critic_opt.step()

            policy_loss = self.be_critic(torch.cat\
                ((states, self.be_actor(states)), dim = 1))
            policy_loss = -torch.mean(policy_loss)
            self.actor_opt.zero_grad()
            policy_loss.backward()
            self.actor_opt.step()

            for be_param, tar_param in zip(self.be_actor.parameters(), \
                self.tar_actor.parameters()):
                tar_param.data.copy_(\
                    tar_param.data * (1 - self.tau) + 
                    be_param.data * self.tau)
            for be_param, tar_param in zip(self.be_critic.parameters(), \
                self.tar_critic.parameters()):
                tar_param.data.copy_(\
                    tar_param.data * (1 - self.tau) + 
                    be_param.data * self.tau)
    
    def save_actor(self):
        torch.save(self.be_actor, self.model_path + 'actor.pt')
    
    def load_actor(self):
        self.be_actor = torch.load(self.model_path + 'actor.pt')







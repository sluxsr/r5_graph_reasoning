# -*- coding: utf-8 -*-
"""
Modified based on the implementation of the policyValueNet in PyTorch by Junxiao Song. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    def __init__(self, width, height):
        super(Net, self).__init__()

        self.width = width
        self.height = height
        # common layers
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*width*height,
                                 width*height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*width*height, 32)
        self.val_fc2 = nn.Linear(32, 1)

        self.init_weights()

    def init_weights(self):
        # torch.nn.init.xavier_uniform(self.conv1.weight)
        # torch.nn.init.xavier_uniform(self.conv2.weight)
        # torch.nn.init.xavier_uniform(self.conv3.weight)

        # torch.nn.init.xavier_uniform(self.act_conv1.weight)
        # torch.nn.init.xavier_uniform(self.act_fc1.weight)

        # torch.nn.init.xavier_uniform(self.val_conv1.weight)
        # torch.nn.init.xavier_uniform(self.val_fc1.weight)
        # torch.nn.init.xavier_uniform(self.val_fc2.weight)

        # self.act_fc1.weight.data.fill_(-0.01)
        pass

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.width*self.height)
        x_act = F.log_softmax(self.act_fc1(x_act))

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.width*self.height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, width, height, use_gpu=True):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.width = width
        self.height = height
        self.l2_const = 1e-4  # coef of l2 penalty
        self.use_gpu = use_gpu

        # the policy value net
        self.policy_value_net = Net(width, height).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

    def policy_value(self, state_batch):
        
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).to(self.device))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, allpair, rules, rules_score):

        legal_positions = allpair.availables

        current_state = np.ascontiguousarray(allpair.current_state(rules, rules_score).reshape(
                -1, 8, self.width, self.height))

        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).to(self.device).float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]

        return act_probs, value

    def train_step(self, state_batch, mcts_probs, z_batch, lr):

        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).to(self.device))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).to(self.device))
            z_batch = Variable(torch.FloatTensor(z_batch).to(self.device))
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            z_batch = Variable(torch.FloatTensor(z_batch))

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)
        log_act_probs, value = self.policy_value_net(state_batch)

        value_loss = F.mse_loss(value.view(-1), z_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()

        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )

        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        net_params = self.get_policy_param()  
        torch.save(net_params, model_file)

    def load_state_dict(self, model_file):
        self.policy_value_net.load_state_dict(model_file)

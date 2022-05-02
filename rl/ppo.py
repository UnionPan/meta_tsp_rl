'''
Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
'''
from running_state import *
from utils import *
from optimizer import conjugate_gradient, line_search
from model import ActorCritic
import math
import datetime
import argparse
import numpy as np
import pandas as pd
from os.path import join as joindir
import os
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from torch.autograd import Variable
from collections import namedtuple
from itertools import count
import scipy.optimize as sciopt
import matplotlib
matplotlib.use('agg')


Transition = namedtuple('Transition', ('state', 'value',
                                       'action', 'logproba', 'mask', 'next_state', 'reward'))

if not os.path.exists('./result'):
    os.mkdir('./result')
EPS = 1e-10
RESULT_DIR = joindir('./result', '.'.join(__file__.split('.')[:-1]))


if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)


class Memory(object):
    "for real just sample the trajectory"
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Turnpike-v0',
                        help='gym environment to test algorithm')
    parser.add_argument('--seed', type=int, default=64,
                        help='random seed')
    parser.add_argument('--num_episode', type=int, default=2000,
                        help='total episode of training')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='batch size of transitions per episode')
    parser.add_argument('--max_step_per_episode', type=int, default=2000,
                        help='maximum step to run per episode')
    parser.add_argument('--gamma', type=float, default=0.995,
                        help='discounted factor')
    parser.add_argument('--lamda', type=float, default=0.97,
                        help='roll out factor')
    parser.add_argument('--log_num_episode', type=int, default=1,
                        help='interval between training status logs (default: 1)')
    parser.add_argument('--num_epoch', type=int, default=10,
                        help='epoch of adjusting approximator weights')
    parser.add_argument('--minibatch_size', type=int, default=256,
                        help='batch size per SGD training')
    parser.add_argument('--clip', type=float, default=0.2,
                        help='clip parameter')
    parser.add_argument('--loss_coeff_value', type=float, default=0.5,
                        help='value loss coefficient')
    parser.add_argument('--loss_coeff_entropy', type=float, default=0.01,
                        help='entropy loss coefficient')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--num_parallel_run', type=int, default=5,
                        help='number of asychronous training process')
    "a bunch of tricks"
    parser.add_argument('--schedule_adam', type=str, default='linear',
                        help='if activate, decay the learning rate at a ratio of (current_episode/total_episode)')
    parser.add_argument('--schedule_clip', type=str, default='linear',
                        help='if activate, decay the clip threshold at a ratio of (current_episode/total_episode)')
    parser.add_argument('--state_norm', type=bool, default=True,
                        help='Normalization of the loss value')
    parser.add_argument('--layer_norm', type=bool, default=True,
                        help='Normalization of the neural layer')
    parser.add_argument('--advantage_norm', type=bool, default=True,
                        help='Normalization of the advantage function')
    parser.add_argument('--lossvalue_norm', type=bool, default=True,
                        help='Normalization of the loss value')
    parser.add_argument('--training', type=bool, default=True,
                        help='choose whether training or testing')

    args = parser.parse_args()
    return args


class PPO:
    def __init__(self, args):
        self.args = args
        self.env = args.env_name
        self.num_inputs = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]

        self.env.seed(self.args.seed)
        torch.manual_seed(args.seed)

        self.network = ActorCritic(
            self.num_inputs, self.num_actions, layer_norm=self.args.layer_norm)
        self.optimizer = opt.Adam(self.network.parameters(), lr=args.lr)

        self.running_state = ZFilter((self.num_inputs,), clip=5.0)

        # record average 1-round cumulative reward in every episode
        self.reward_record = []
        self.global_steps = 0
        self.reward_plot = []

        self.lr_now = self.args.lr
        self.clip_now = self.args.clip

    def collect_advantage_func(self, memory, **kargs):
        batch = memory.sample()
        batch_size = len(memory)

        # step2: extract variables from trajectories
        rewards = Tensor(batch.reward)
        values = Tensor(batch.value)
        masks = Tensor(batch.mask)
        actions = Tensor(batch.action)
        states = Tensor(batch.state)
        oldlogproba = Tensor(batch.logproba)

        returns = Tensor(batch_size)
        deltas = Tensor(batch_size)
        advantages = Tensor(batch_size)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + self.args.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.args.gamma * \
                prev_value * masks[i] - values[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            advantages[i] = deltas[i] + self.args.gamma * \
                self.args.lamda * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
        if self.args.advantage_norm:
            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + EPS)

        return advantages, returns

    def clipped_training(self, advantages, memory, returns_computed, i_episode):
        batch = memory.sample()
        batch_size = len(memory)
        minibatch_size = self.args.minibatch_size

        rewards = Tensor(batch.reward)
        values = Tensor(batch.value)
        masks = Tensor(batch.mask)
        actions = Tensor(batch.action)
        states = Tensor(batch.state)
        oldlogproba = Tensor(batch.logproba)
        returns = returns_computed

        for i_epoch in range(int(self.args.num_epoch * batch_size / self.args.minibatch_size)):
            # sample from current batch
            minibatch_ind = np.random.choice(
                batch_size, minibatch_size, replace=False)
            minibatch_states = states[minibatch_ind]
            minibatch_actions = actions[minibatch_ind]
            minibatch_oldlogproba = oldlogproba[minibatch_ind]
            minibatch_newlogproba = self.network.get_logproba(
                minibatch_states, minibatch_actions)
            minibatch_advantages = advantages[minibatch_ind]
            minibatch_returns = returns[minibatch_ind]
            minibatch_newvalues = self.network._forward_critic(
                minibatch_states).flatten()

            ratio = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
            surr1 = ratio * minibatch_advantages
            surr2 = ratio.clamp(1 - self.clip_now, 1 +
                                self.clip_now) * minibatch_advantages
            loss_surr = - torch.mean(torch.min(surr1, surr2))

            if self.args.lossvalue_norm:
                minibatch_return_6std = 6 * minibatch_returns.std()
                loss_value = torch.mean(
                    (minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
            else:
                loss_value = torch.mean(
                    (minibatch_newvalues - minibatch_returns).pow(2))

            loss_entropy = torch.mean(
                torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

            total_loss = loss_surr + self.args.loss_coeff_value * \
                loss_value + self.args.loss_coeff_entropy * loss_entropy
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        if self.args.schedule_clip == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            clip_now = self.args.clip * ep_ratio

        if self.args.schedule_adam == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            lr_now = self.args.lr * ep_ratio
            for g in self.optimizer.param_groups:
                g['lr'] = lr_now

        if i_episode % args.log_num_episode == 0:
            print('Finished episode: {} \
                   Reward: {:.4f} \
                   total_loss = {:.4f} = {:.4f} + {} * {:.4f} + {} * {:.4f}'
                  .format(i_episode,
                          self.reward_record[-1]['meanepreward'],
                          total_loss.data,
                          loss_surr.data,
                          self.args.loss_coeff_value,
                          loss_value.data, self.args.loss_coeff_entropy, loss_entropy.data))
            print('-----------------')

    def collect_trajectories(self, i_episode):
        num_steps = 0
        reward_list = []
        len_list = []
        memory = Memory()

        while num_steps < self.args.batch_size:
            state = self.env.reset()
            if self.args.state_norm:
                state = self.running_state(state)
            reward_sum = 0
            for t in range(self.args.max_step_per_episode):

                action_mean, action_logstd, value = self.network(
                    Tensor(state).unsqueeze(0))
                action, logproba = self.network.select_action(
                    action_mean, action_logstd)
                action = action.data.numpy()[0]
                logproba = logproba.data.numpy()[0]
                next_state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                if self.args.state_norm:
                    next_state = self.running_state(next_state)
                mask = 0 if done else 1

                memory.push(state, value, action, logproba,
                            mask, next_state, reward)

                if done:
                    break
                state = next_state

            num_steps += (t + 1)
            self.global_steps += (t + 1)

            reward_list.append(reward_sum)
            len_list.append(t + 1)

            self.reward_plot.append(
                {'steps': self.global_steps, 'reward': reward_sum})
        self.reward_record.append({
            'episode': i_episode,
            'steps': self.global_steps,
            'meanepreward': np.mean(reward_list),
            'meaneplen': np.mean(len_list)})

        return memory

    def ppo_step(self, j_parallel):
        self.global_steps = 0
        for i_episode in range(self.args.num_episode):
            # collect trajectories using current policy    on-policy training
            memory = self.collect_trajectories(i_episode)
            # collect advantage function and returns
            # ref: https://arxiv.org/abs/1506.02438
            advantages, returns = self.collect_advantage_func(memory)
            # training the actor-critic network
            self.clipped_training(advantages, memory, returns, i_episode)

        torch.save(self.network.state_dict(
        ), './saved_model/ppo_ac_{}_{}'.format(self.args.env_name, j_parallel))


    def train_turnpike(self):
        """training a single environment of new Jersey Turnpike"""
        datestr = datetime.datetime.now().strftime('%Y-%m-%d')

        record_dfs = pd.DataFrame(columns=['steps', 'reward'])
        reward_cols = []

        print("****--***** training turnpike environment ****--*****") 
        for j in range(self.args.num_parallel_run):
            self.args.seed += 1
            self.reward_record = []
            self.reward_plot = []
            self.global_steps = 0
            self.ppo_step(j)
            reward_record_temp = pd.DataFrame(self.reward_plot)
            record_dfs = record_dfs.merge(reward_record_temp, how = 'outer', on='steps', suffixes=('', '_{}'.format(j)))
            reward_cols.append('reward_{}'.format(j))

        self.plot_figure(datestr, record_dfs, reward_cols)

    def plot_figure(self, datestamp, record_dfs, reward_cols):
        "ploting figure using pandas"

        record_dfs = record_dfs.drop(columns='reward').sort_values(
            by='steps', ascending=True).ffill().bfill()
        record_dfs['reward_mean'] = record_dfs[reward_cols].mean(axis=1)
        record_dfs['reward_std'] = record_dfs[reward_cols].std(axis=1)
        record_dfs['reward_smooth'] = record_dfs['reward_mean'].ewm(
            span=20).mean()
        record_dfs.to_csv(joindir(
            RESULT_DIR, 'ppo-record-{}-{}.csv'.format(self.args.env_name, datestamp)))

    # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(record_dfs['steps'],
                 record_dfs['reward_mean'], label='trajory reward')
        plt.plot(record_dfs['steps'],
                 record_dfs['reward_smooth'], label='smoothed reward')
        plt.fill_between(record_dfs['steps'], record_dfs['reward_mean'] - record_dfs['reward_std'],
                         record_dfs['reward_mean'] + record_dfs['reward_std'], color='b', alpha=0.2)
        plt.legend()
        plt.xlabel('global steps')
        plt.ylabel('average reward')
        plt.title('PPO on {}'.format(self.args.env_name))
        plt.savefig(
            joindir(RESULT_DIR, 'ppo-{}-{}.pdf'.format(self.args.env_name, datestamp)))

    def render_test(self, env):
        "testing a specific environment using current loaded policy"

        args.env_name = env
        self.__init__(args)
        self.network.load_state_dict(torch.load(
            './saved_model/ppo_ac_{}_{}'.format(env, 2)))
        self.network.eval()
        torch.no_grad()

        state = self.env.reset()
        for t in range(10000):
            self.env.render()
            if self.args.state_norm:
                state = self.running_state(state)
            action_mean, action_logstd = self.network._forward_actor(
                Tensor(state).unsqueeze(0))
            action, logproba = self.network.select_action(
                action_mean, action_logstd)
            action = action.data.numpy()[0]
            next_state, reward, done, _ = self.env.step(action)
            state = next_state


if __name__ == "__main__":

    args = add_arguments()
    ppo_algo = PPO(args)
    #ppo_algo.ppo_step(0)
    #ppo_algo.test_benchmarks()
    #ppo_algo.render_test('Humanoid-v2')

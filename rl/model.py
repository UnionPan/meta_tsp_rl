import datetime
import os
from collections import namedtuple
import math
from torch.autograd import Variable
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.autograd as autograd
import torch
from torch.distributions import Categorical


Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))
OldCopy = namedtuple(
    'OldCopy', ('log_density', 'action_mean', 'action_log_std', 'action_std'))



class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    @staticmethod
    def _normal_log_density(x, mean, log_std, std):
        """
        returns probability distribution of normal distribution N(x; mean, std)
        :param x, mean, log_std, std: torch.Tensor
        :return: log probability density torch.Tensor
        """
        var = std.pow(2)
        log_density = - 0.5 * \
            math.log(2 * math.pi) - log_std - (x - mean).pow(2) / (2 * var)
        return log_density.sum(1)

    def _get_log_p_a_s(self, states, actions, return_net_output=False):
        """
        get log p(a|s) on data (states, actions)
        :param states, actions: torch.Tensor
        :return: log probability density torch.Tensor 
        """
        action_means, action_log_stds, action_stds = self.__call__(states)
        log_density = self._normal_log_density(
            actions, action_means, action_log_stds, action_stds)
        if return_net_output:
            return OldCopy(log_density=Variable(log_density),
                           action_mean=Variable(action_means),
                           action_log_std=Variable(action_log_stds),
                           action_std=Variable(action_stds))
        else:
            return log_density

    def set_old_loss(self, states, actions):
        self.old_copy = self._get_log_p_a_s(
            states, actions, return_net_output=True)

    def get_loss(self, states, actions, advantages):
        """
        get loss variable
        loss = dfrac{pi_theta (a|s)}{q(a|s)} Q(s, a)
        :param states: torch.Tensor
        :param actions: torch.Tensor
        :param advantages: torch.Tensor
        :return: the loss, torch.Variable
        """
        assert self.old_copy is not None
        log_prob = self._get_log_p_a_s(states, actions)
        # notice Variable(x) here means x is treated as fixed data
        # and autograd is not applied to parameters that generated x.
        # in another word, pi_{theta_old}(a|s) is fixed and the gradient is taken w.r.t. new theta
        action_loss = - advantages * \
            torch.exp(log_prob - self.old_copy.log_density)
        return action_loss.mean()

    def get_kl(self, states):
        """
        given old and new (mean, log_std, std) calculate KL divergence 
        pay attention 
            1. the distribution is a normal distribution on a continuous domain
            2. the KL divergence is a integration over (-inf, inf) 
                KL = integrate p0(x) log(p0(x) / p(x)) dx
        thus, KL can be calculated analytically
                KL = log_std - log_std0 + (var0 + (mean - mean0)^2) / (2 var) - 1/2
        ref: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        :param states: torch.Tensor(#samples, #d_state)
        :return: KL torch.Tensor(1)
        """
        action_mean, action_log_std, action_std = self.__call__(states)
        kl = action_log_std - self.old_copy.action_log_std \
            + (self.old_copy.action_std.pow(2) + (self.old_copy.action_mean - action_mean).pow(2)) \
            / (2.0 * action_std.pow(2)) - 0.5
        return kl.sum(1).mean()

    def kl_hessian_times_vector(self, states, v):
        """
        return the product of KL's hessian and an arbitrary vector in O(n) time
        ref: https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
        :param states: torch.Tensor(#samples, #d_state) used to calculate KL divergence on samples
        :param v: the arbitrary vector, torch.Tensor
        :return: (H + damping * I) dot v, where H = nabla nabla KL
        """
        kl = self.get_kl(states)
        # here, set create_graph=True to enable second derivative on function of this derivative
        grad_kl = torch.autograd.grad(kl, self.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grad_kl])

        grad_kl_v = (flat_grad_kl * v).sum()
        grad_grad_kl_v = torch.autograd.grad(grad_kl_v, self.parameters())
        flat_grad_grad_kl_v = torch.cat(
            [grad.contiguous().view(-1) for grad in grad_grad_kl_v])

        return flat_grad_grad_kl_v + args.damping * v

    def set_flat_params(self, flat_params):
        """
        set flat_params
        : param flat_params: Tensor
        """
        flat_params = Tensor(flat_params)
        prev_ind = 0
        for param in self.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size
        self.old_log_prob = None

    def get_flat_params(self):
        """
        get flat parameters
        returns numpy array
        """
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params.double().numpy()

    def save_model_policy(self, args):
        """
        save model
        """
        if not os.path.exists('./saved_model'):
            os.makedirs('./saved_model')
        np.save('./saved_model/param_policy_{}'.format(args.env_name),
                self.get_flat_params())

    def load_model_policy(self, args):
        flat_params = np.load(
            './saved_model/param_policy_{}.npy'.format(args.env_name))
        self.set_flat_params(flat_params)


class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values.squeeze()

    def get_flat_params(self):
        """
        get flat parameters
        
        :return: flat param, numpy array
        """
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params.double().numpy()

    def get_flat_grad(self):
        """
        get flat gradient
        
        :return: flat grad, numpy array
        """
        grads = []
        for param in self.parameters():
            grads.append(param.grad.view(-1))

        flat_grad = torch.cat(grads)
        return flat_grad.double().numpy()

    def set_flat_params(self, flat_params):
        """
        set flat_params
        
        :param flat_params: numpy.ndarray
        """
        flat_params = Tensor(flat_params)
        prev_ind = 0
        for param in self.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size

    def reset_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

    def get_sum_squared_params(self):
        """
        sum of squared parameters used for L2 regularization
        returns a Variable
        """
        ans = Variable(Tensor([0]))
        for param in self.parameters():
            ans += param.pow(2).mean()
        return ans

    def save_model_value(self, args):
        """
        save model
        """
        if not os.path.exists('./saved_model'):
            os.makedirs('./saved_model')

        np.save('./saved_model/param_value_{}'.format(args.env_name),
                self.get_flat_params())

    def load_model_value(self, args):
        flat_params = np.load(
            './saved_model/param_value_{}.npy'.format(args.env_name))
        self.set_flat_params(flat_params)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, layer_norm=True):
        super(ActorCritic, self).__init__()

        self.actor_fc1 = nn.Linear(num_inputs, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, num_outputs)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))

        self.critic_fc1 = nn.Linear(num_inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        if layer_norm:
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3, std=0.01)

            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            self.layer_norm(self.critic_fc3, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states
        :return: 3 Tensor2
        """
        action_mean, action_logstd = self._forward_actor(states)
        critic_value = self._forward_critic(states)
        return action_mean, action_logstd, critic_value

    def _forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        action_mean = self.actor_fc3(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def _forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value

    def select_action(self, action_mean, action_logstd, return_logproba=True):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        """
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        if return_logproba:
            logproba = self._normal_logproba(
                action, action_mean, action_logstd, action_std)
        return action, logproba

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        var = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - \
            logstd - (x - mean).pow(2) / (2 * var)
        return logproba.sum(1)

    def get_logproba(self, states, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, action_logstd = self._forward_actor(states)
        logproba = self._normal_logproba(actions, action_mean, action_logstd)
        return logproba


class DiscreteAC(nn.Module):
    "A discrete action Actor Critic class"
    def __init__(self, num_inputs, num_outputs, layer_norm=True):
        super(DiscreteAC, self).__init__()

        self.actor_fc1 = nn.Linear(num_inputs, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, num_outputs)

        self.critic_fc1 = nn.Linear(num_inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        if layer_norm:
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3, std=0.01)

            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            self.layer_norm(self.critic_fc3, std=1.0)
    
    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    
    def forward(self, states):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states
        :return: 3 Tensor2
        """
        pi_a = self._forward_actor(states)
        critic_value = self._forward_critic(states)
        return pi_a, critic_value
    
    def _forward_actor(self, states):
        #print(states.shape)
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        logits = self.actor_fc3(x)
        return Categorical(logits=logits)

    def _forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value
    
    def select_action(self, pi_a, return_logproba=True):
        action = pi_a.sample()
        logprob = pi_a.log_prob(action)

        return action, logprob


    @staticmethod
    def _logit_logproba(x, pis):
        assert pis.logits.shape[0] == x.shape[0]
        probs = Tensor([pis.probs[i, int(x[i])] for i in range(x.shape[0])])
        logprobs = torch.log(probs)
        return logprobs

    def get_logproba(self, states, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        pis = self._forward_actor(states)
        logproba = self._logit_logproba(actions, pis)
        return logproba


if  __name__ == "__main__":
    ac = DiscreteAC(3, 3)
    states = Tensor([[0.2, 1.3, 0.4], [0.3, 0.2,2.1], [0.9, 0.8, 0.7]])
    actions = Tensor([[1.0], [2.0], [0.0]])
    print(ac.get_logproba(states, actions))

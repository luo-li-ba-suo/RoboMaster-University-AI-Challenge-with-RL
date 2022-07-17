import torch
import torch.nn as nn
import numpy as np

if torch.cuda.is_available():
    print('GPU is available')
else:
    print('GPU cannot be used')


class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim), )
        layer_norm(self.net[-1], std=0.1)  # output layer for action

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_logstd = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state):
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_logprob_entropy(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        delta = noise.pow(2) * 0.5
        return -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)  # old_logprob


class ActorDiscretePPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.Multi_Discrete = True

        self.main_net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.action_nets = nn.Sequential(*[nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                                         nn.Linear(mid_dim, action_d)) for action_d in action_dim])
        # self.action_nets = nn.Sequential(*[nn.Sequential(nn.Linear(mid_dim, action_d)) for action_d in action_dim])
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, state):
        hidden = self.main_net(state)
        return torch.cat([net(hidden) for net in self.action_nets], dim=1)  # action_prob without softmax

    def get_action(self, state):
        result = self.forward(state)
        if not self.Multi_Discrete:
            a_prob = self.soft_max(result)
            # dist = Categorical(a_prob)
            # action = dist.sample()
            samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
            action = samples_2d.reshape(state.size(0))
        else:
            a_prob = []
            action = []
            n = 0
            for action_dim_ in self.action_dim:
                a_prob_ = self.soft_max(result[:, n:n + action_dim_])
                a_prob.append(a_prob_)
                n += action_dim_
                samples_2d = torch.multinomial(a_prob_, num_samples=1, replacement=True)
                action_ = samples_2d.reshape(state.size(0))
                action.append(action_)
        return action, a_prob

    def get_max_action(self, state):
        result = self.forward(state)
        if self.Multi_Discrete:
            n = 0
            action = []
            for action_dim_ in self.action_dim:
                action.append(result[:, n:n + action_dim_].argmax(dim=1).detach().cpu().numpy()[0])
                n += action_dim_
        return action

    def get_logprob_entropy(self, state, action):
        result = self.forward(state)
        if self.Multi_Discrete:
            a_prob = []
            dist_prob = []
            dist_entropy = []
            n = 0
            for i, action_dim_ in enumerate(self.action_dim):
                a_prob_ = self.soft_max(result[:, n:n + action_dim_])
                a_prob.append(a_prob_)
                dist = self.Categorical(a_prob_)
                dist_prob.append(dist.log_prob(action[:, i].long()))
                dist_entropy.append(dist.entropy().mean())
                n += action_dim_
            return sum(dist_prob), sum(dist_entropy) / len(dist_entropy)

        else:
            a_prob = self.soft_max(self.net(state))
            dist = self.Categorical(a_prob)
            a_int = action.squeeze(1).long()
            return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, action, a_prob):
        if self.Multi_Discrete:
            n = 0
            dist_log_prob = []
            for i, action_dim_ in enumerate(self.action_dim):
                # try:
                dist_log_prob.append(self.Categorical(a_prob[:, n: n + action_dim_]).log_prob(action[:, i].long()))
                # except ValueError as _:
                #     for i, prob in enumerate(a_prob[:, n: n + action_dim_]):
                #         try:
                #             _ = self.Categorical(prob)
                #         except ValueError as e:
                #             print(e)
                #             print(i, ' : ', a_prob[i])
                n += action_dim_
            return sum(dist_log_prob)
        else:
            dist = self.Categorical(a_prob)
            return dist.log_prob(action.long().squeeze(1))


class MultiAgentActorDiscretePPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.Multi_Discrete = True

        self.main_net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.action_nets = nn.Sequential(*[nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                                         nn.Linear(mid_dim, action_d)) for action_d in action_dim])
        for net in self.action_nets:
            layer_norm(net[-1], std=0.01)
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, state):
        hidden = self.main_net(state)
        return torch.cat([net(hidden) for net in self.action_nets], dim=1)  # action_prob without softmax

    def get_action(self, state, stochastic=0, deterministic=0):
        result = self.forward(state)
        stochastic_action = a_prob = deterministic_action = None
        if stochastic is 0 and deterministic is 0:
            stochastic_action = a_prob = None
        else:
            if stochastic:
                result_stochastic = result[0:stochastic]
                stochastic_action, a_prob = self.get_stochastic_action(result_stochastic)
            if deterministic:
                start = 0 if not stochastic else stochastic
                result_deterministic = result[start:deterministic + start]
                deterministic_action = self.get_deterministic_action(result_deterministic)

        return stochastic_action, a_prob, deterministic_action

    def get_stochastic_action(self, result):
        if self.Multi_Discrete:
            a_prob = []
            action = []
            n = 0
            for action_dim_ in self.action_dim:
                a_prob_ = self.soft_max(result[:, n:n + action_dim_])
                a_prob.append(a_prob_)
                n += action_dim_
                samples_2d = torch.multinomial(a_prob_, num_samples=1, replacement=True)
                action_ = samples_2d.reshape(result.size(0))
                action.append(action_)
        else:
            a_prob = self.soft_max(result)
            samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
            action = samples_2d.reshape(result.size(0))
        return action, a_prob

    def get_deterministic_action(self, result):
        if self.Multi_Discrete:
            n = 0
            action = []
            for action_dim_ in self.action_dim:
                action.append(result[:, n:n + action_dim_].argmax(dim=1).detach())
                n += action_dim_
        else:
            raise NotImplementedError
        return action

    def get_logprob_entropy(self, state, action):
        result = self.forward(state)
        if self.Multi_Discrete:
            a_prob = []
            dist_prob = []
            dist_entropy = []
            n = 0
            for i, action_dim_ in enumerate(self.action_dim):
                a_prob_ = self.soft_max(result[:, n:n + action_dim_])
                a_prob.append(a_prob_)
                dist = self.Categorical(a_prob_)
                dist_prob.append(dist.log_prob(action[:, i].long()))
                dist_entropy.append(dist.entropy().mean())
                n += action_dim_
            return sum(dist_prob), sum(dist_entropy) / len(dist_entropy)

        else:
            a_prob = self.soft_max(self.net(state))
            dist = self.Categorical(a_prob)
            a_int = action.squeeze(1).long()
            return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, action, a_prob):
        if self.Multi_Discrete:
            n = 0
            dist_log_prob = []
            for i, action_dim_ in enumerate(self.action_dim):
                # try:
                dist_log_prob.append(self.Categorical(a_prob[:, n: n + action_dim_]).log_prob(action[:, i].long()))
                # except ValueError as _:
                #     for i, prob in enumerate(a_prob[:, n: n + action_dim_]):
                #         try:
                #             _ = self.Categorical(prob)
                #         except ValueError as e:
                #             print(e)
                #             print(i, ' : ', a_prob[i])
                n += action_dim_
            return sum(dist_log_prob)
        else:
            dist = self.Categorical(a_prob)
            return dist.log_prob(action.long().squeeze(1))


class CriticAdv(nn.Module):
    def __init__(self, mid_dim, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))
        layer_norm(self.net[-1], std=0.5)  # output layer for V value

    def forward(self, state):
        return self.net(state)  # V value


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

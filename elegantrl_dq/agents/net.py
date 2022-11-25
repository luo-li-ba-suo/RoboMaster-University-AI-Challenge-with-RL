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
    def __init__(self, mid_dim, state_dim, action_dim, if_use_cnn=False, state_cnn_channel=6, if_use_rnn=False,
                 if_use_conv1D=True, state_seq_len=30):
        super().__init__()
        self.action_dim = action_dim
        self.Multi_Discrete = True
        self.if_use_cnn = if_use_cnn
        self.if_use_conv1D = if_use_conv1D
        self.cnn_out_dim = 64
        self.conv1D_kernel_size = 3
        self.vector_net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU())
        if if_use_conv1D:
            self.conv_net = nn.Sequential(nn.Conv1d(state_cnn_channel, 32, self.conv1D_kernel_size),
                                          nn.ReLU(), nn.Flatten())
            self.cnn_out_dim = (state_seq_len + (self.conv1D_kernel_size//2)*2 - self.conv1D_kernel_size + 1) * 32
        elif if_use_cnn:
            self.conv_net = nn.Sequential(nn.Conv2d(state_cnn_channel, 16, 3),
                                              nn.ReLU(),
                                              nn.Conv2d(16, 32, 3, stride=2),
                                              nn.ReLU(),
                                              nn.Conv2d(32, self.cnn_out_dim, 3, stride=2),
                                              nn.ReLU(),
                                              nn.AdaptiveMaxPool2d(1),
                                              nn.Flatten())
        else:
            self.cnn_out_dim = 0
        self.hidden_net = nn.Sequential(nn.Linear(mid_dim + self.cnn_out_dim, mid_dim), nn.ReLU())
        self.rnn = None
        self.action_nets = nn.Sequential(*[nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                                         nn.Linear(mid_dim, action_d)) for action_d in action_dim])
        for net in self.action_nets:
            layer_norm(net[-1], std=0.01)
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, state, state_cnn=None, rnn_state=None):
        hidden = self.vector_net(state)
        if self.if_use_conv1D:
            state_cnn = state_cnn.view(-1, *state_cnn.shape[-2:])
            state_cnn = torch.cat([state_cnn[:, :, :(self.conv1D_kernel_size // 2)], state_cnn,
                                   state_cnn[:, :, -(self.conv1D_kernel_size // 2):]], dim=-1)
        if self.if_use_cnn or self.if_use_conv1D:
            CNN_out = self.conv_net(state_cnn)
            hidden = self.hidden_net(torch.cat((hidden, CNN_out), dim=1))
        else:
            hidden = self.hidden_net(hidden)
        return torch.cat([net(hidden) for net in self.action_nets], dim=1)  # action_prob without softmax

    def get_stochastic_action(self, state, state_cnn=None, rnn_state=None):
        result = self.forward(state, state_cnn, rnn_state)
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
        return action, a_prob, None

    def get_deterministic_action(self, state, state_cnn=None, rnn_state=None):
        result = self.forward(state, state_cnn, rnn_state)
        n = 0
        action = []
        for action_dim_ in self.action_dim:
            action.append(result[:, n:n + action_dim_].argmax(dim=1).detach())
            n += action_dim_
        return action, None

    def get_logprob_entropy(self, state, action, state_cnn=None, state_rnn=None):
        result = self.forward(state, state_cnn, state_rnn)
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

    def get_old_logprob(self, action, a_prob):
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


class CriticAdv(nn.Module):
    def __init__(self, mid_dim, state_dim, if_use_cnn=False, state_cnn_channel=6,
                 if_use_conv1D=True, state_seq_len=30):
        super().__init__()
        self.if_use_cnn = if_use_cnn
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU())
        self.cnn_out_dim = 64
        self.if_use_conv1D = if_use_conv1D
        self.conv1D_kernel_size = 3
        if if_use_conv1D:
            self.conv_net = nn.Sequential(nn.Conv1d(state_cnn_channel, 32, self.conv1D_kernel_size),
                                          nn.ReLU(), nn.Flatten())
            self.cnn_out_dim = (state_seq_len + (self.conv1D_kernel_size//2)*2 - self.conv1D_kernel_size + 1) * 32
        elif if_use_cnn:
            self.conv_net = nn.Sequential(nn.Conv2d(state_cnn_channel, 16, 3),
                                                  nn.ReLU(),
                                                  nn.Conv2d(16, 32, 3, stride=2),
                                                  nn.ReLU(),
                                                  nn.Conv2d(32, 64, 3, stride=2),
                                                  nn.ReLU(),
                                                  nn.AdaptiveMaxPool2d(1),
                                                  nn.Flatten())
        else:
            self.conv_net = None
            self.cnn_out_dim = 0
        self.hidden_net = nn.Sequential(nn.Linear(mid_dim + self.cnn_out_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, 1))
        layer_norm(self.hidden_net[-1], std=0.5)  # output layer for V value

    def forward(self, state, state_cnn=None, rnn_state=None):
        hidden = self.net(state)
        if self.if_use_conv1D:
            state_cnn = state_cnn.view(-1, *state_cnn.shape[-2:])
            state_cnn = torch.cat([state_cnn[:, :, :(self.conv1D_kernel_size // 2)], state_cnn,
                                   state_cnn[:, :, -(self.conv1D_kernel_size // 2):]], dim=-1)
        if self.if_use_cnn or self.if_use_conv1D:
            CNN_out = self.conv_net(state_cnn)
            hidden = self.hidden_net(torch.cat((hidden, CNN_out), dim=1))
        else:
            hidden = self.hidden_net(hidden)
        return hidden


# actor与critic网络共享特征提取的主干
class DiscretePPOShareNet(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_cnn=False, if_use_conv1D=True,
                 if_use_rnn=False, rnn_state_size=128, LSTM_or_GRU=True,
                 state_cnn_channel=6, state_seq_len=30, actor_obs_dim=None):
        super().__init__()
        '''参数处理'''
        self.action_dim = action_dim  # 默认为多维离散动作空间
        self.actor_obs_dim = actor_obs_dim
        if self.actor_obs_dim is None:
            self.actor_obs_dim = state_dim
        if self.actor_obs_dim < state_dim:
            self.use_extra_obs_critic = True
        else:
            self.use_extra_obs_critic = False
        '''Recurrent Network'''
        self.if_use_rnn = if_use_rnn
        rnn_state_size = rnn_state_size
        self.LSTM_or_GRU = LSTM_or_GRU

        '''1.共享特征提取层：'''
        '''观测向量的处理'''
        self.vector_net = nn.Sequential(nn.Linear(self.actor_obs_dim, mid_dim), nn.ReLU())
        self.extra_obs_net = None
        if self.use_extra_obs_critic:
            self.extra_obs_net = nn.Sequential(nn.Linear(state_dim - self.actor_obs_dim, mid_dim), nn.ReLU())
        '''观测矩阵的处理'''
        self.if_use_cnn = if_use_cnn
        self.if_use_conv1D = if_use_conv1D
        if if_use_conv1D:
            self.conv1D_kernel_size = 3
            self.conv_net = nn.Sequential(nn.Conv1d(state_cnn_channel, 32, self.conv1D_kernel_size),
                                          nn.ReLU(), nn.Flatten())
            cnn_out_dim = (state_seq_len + (self.conv1D_kernel_size//2)*2 - self.conv1D_kernel_size + 1) * 32
        elif if_use_cnn:
            cnn_out_dim = 64
            self.conv_net = nn.Sequential(nn.Conv2d(state_cnn_channel, 16, 3),
                                              nn.ReLU(),
                                              nn.Conv2d(16, 32, 3, stride=2),
                                              nn.ReLU(),
                                              nn.Conv2d(32, cnn_out_dim, 3, stride=2),
                                              nn.ReLU(),
                                              nn.AdaptiveMaxPool2d(1),
                                              nn.Flatten())
        else:
            self.conv_net = None
            cnn_out_dim = 0
        '''2.共享中间层'''
        self.hidden_net = nn.Sequential(nn.Linear(mid_dim + cnn_out_dim, mid_dim), nn.ReLU())
        '''3.共享记忆层'''
        if self.if_use_rnn:
            if self.LSTM_or_GRU:
                self.rnn = nn.LSTM(mid_dim, rnn_state_size)
            else:
                self.rnn = nn.GRU(mid_dim, rnn_state_size)
            for name, param in self.rnn.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)
        '''4.策略层'''
        input_dim = rnn_state_size if self.if_use_rnn else mid_dim
        self.action_nets = nn.Sequential(*[nn.Sequential(nn.Linear(input_dim, mid_dim), nn.ReLU(),
                                                         nn.Linear(mid_dim, action_d)) for action_d in action_dim])
        for net in self.action_nets:
            layer_norm(net[-1], std=0.01)
        self.forward = self.actor
        '''5.价值层'''
        input_dim = rnn_state_size if self.if_use_rnn else mid_dim
        input_dim += mid_dim if self.use_extra_obs_critic else 0
        self.value_net = nn.Sequential(nn.Linear(input_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, 1))
        layer_norm(self.value_net[-1], std=0.5)

        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def share_net(self, state, state_cnn=None, rnn_state=None):
        hidden = self.vector_net(state)
        if self.if_use_conv1D:
            state_cnn = state_cnn.view(-1, *state_cnn.shape[-2:])
            state_cnn = torch.cat([state_cnn[:, :, :(self.conv1D_kernel_size // 2)], state_cnn,
                                   state_cnn[:, :, -(self.conv1D_kernel_size // 2):]], dim=-1)
        if self.if_use_cnn or self.if_use_conv1D:
            CNN_out = self.conv_net(state_cnn)
            hidden = torch.cat((hidden, CNN_out), dim=1)
        hidden = self.hidden_net(hidden)
        if self.if_use_rnn:
            if self.LSTM_or_GRU:
                hidden, rnn_state = self.rnn(hidden.unsqueeze(0), (rnn_state[0].unsqueeze(0), rnn_state[1].unsqueeze(0)))
                rnn_state = [s.squeeze(0) for s in rnn_state]
            else:
                hidden, rnn_state = self.rnn(hidden.unsqueeze(0), rnn_state[0].unsqueeze(0))
            hidden = hidden.squeeze(0)
        return hidden, rnn_state

    def actor(self, state, state_cnn=None, rnn_state=None):
        hidden, rnn_state = self.share_net(state, state_cnn, rnn_state)
        return torch.cat([net(hidden) for net in self.action_nets], dim=1), rnn_state  # action_prob without softmax

    def critic(self, state, state_cnn=None, rnn_state=None):
        # 注意这里的state相对self.actor中的state可能多出一部分：extra_obs
        hidden, _ = self.share_net(state[..., :self.actor_obs_dim], state_cnn, rnn_state)
        if self.use_extra_obs_critic:
            extra_hidden = self.extra_obs_net(state[..., self.actor_obs_dim:])
            hidden = torch.cat((hidden, extra_hidden), dim=1)
        return self.value_net(hidden)

    def get_stochastic_action(self, state, state_cnn=None, rnn_state=None):
        result, rnn_state = self.actor(state, state_cnn, rnn_state)
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
        return action, a_prob, rnn_state

    def get_deterministic_action(self, state, state_cnn=None, rnn_state=None):
        result, rnn_state = self.actor(state, state_cnn, rnn_state)
        n = 0
        action = []
        for action_dim_ in self.action_dim:
            action.append(result[:, n:n + action_dim_].argmax(dim=1).detach())
            n += action_dim_
        return action, rnn_state

    def get_logprob_entropy(self, state, action, state_cnn=None, state_rnn=None):
        result, _ = self.actor(state, state_cnn, state_rnn)
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

    def get_old_logprob(self, action, a_prob):
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


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

import os
import sys
import time
from copy import deepcopy
from pathlib import Path
import gym

gym.logger.set_level(40)
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
import simulator
import wandb

if torch.cuda.is_available():
    print('GPU is available')
else:
    print('GPU cannot be used')
"""
cancel prepare_buffer
move env.last_state to buffer.last_state
"""

"""net.py"""


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


"""agent.py"""


class AgentPPO:
    def __init__(self):
        super().__init__()
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.02
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None

        self.states = None
        self.device = None
        self.criterion = None
        self.act = self.enemy_act = self.act_optimizer = None
        self.cri = self.cri_optimizer = self.cri_target = None
        self.trainer_id = 0

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

        self.act = ActorPPO(net_dim, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(net_dim, state_dim).to(self.device)
        self.cri_target = deepcopy(self.cri) if self.cri_target is True else self.cri

        self.criterion = torch.nn.SmoothL1Loss()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)

    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(states)  # plan to be get_action_a_noise
        if isinstance(actions, list):  # 如果是Multi-Discrete
            return np.array([action[0].detach().cpu().numpy() for action in actions]), [noise[0].detach().cpu().numpy()
                                                                                        for noise in noises]
        else:
            return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = list()

        state = self.states
        for _ in range(target_step):
            env.render()
            action, noise = self.select_action(state)
            next_state, reward, done, _ = env.step(np.tanh(action))

            other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
            trajectory_list.append((state, other))

            state = env.reset() if done else next_state
        self.states = state
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times, repeat_times_policy, soft_update_tau, if_train_actor=True):
        buffer.update_now_len()
        buf_len = buffer.now_len
        buf_state, buf_action, buf_r_sum, buf_logprob, buf_advantage = self.prepare_buffer(buffer)
        buffer.empty_buffer()

        update_policy_net = int(buf_len / batch_size) * repeat_times_policy  # 策略更新只利用一次数据

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = logprob = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            if update_policy_net > 0:
                new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = advantage * ratio
                surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
                if if_train_actor:
                    self.optim_update(self.act_optimizer, obj_actor)
                update_policy_net -= 1

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        return obj_critic.item(), obj_actor.item(), logprob.mean().item()  # logging_tuple

    def prepare_buffer(self, buffer):
        buf_len = buffer.now_len

        with torch.no_grad():  # compute reverse reward
            reward, mask, action, a_noise, state = buffer.sample_all()

            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            value = torch.cat([self.cri_target(state[i:i + bs]) for i in range(0, state.size(0), bs)], dim=0)
            logprob = self.act.get_old_logprob(action, a_noise)
            # self.states[0]表示第一個智能體的狀態
            pre_state = torch.as_tensor((self.states[self.trainer_id],), dtype=torch.float32, device=self.device)
            pre_r_sum = self.cri(pre_state).detach()
            r_sum, advantage = self.get_reward_sum(self, buf_len, reward, mask, value, pre_r_sum)
        return state, action, r_sum, logprob, advantage

    @staticmethod
    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value, pre_r_sum) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    @staticmethod
    def get_reward_sum_gae(self, buf_len, buf_reward, buf_mask, buf_value, pre_r_sum) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + buf_mask[i] * (pre_advantage - buf_value[i])  # fix a bug here
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))

    def save_load_model(self, cwd, if_save):
        """save or load model files

        :str cwd: current working directory, we save model file here
        :bool if_save: save model or load model
        """
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if self.act is not None:
                torch.save(self.act.state_dict(), act_save_path)
            if self.cri is not None:
                torch.save(self.cri.state_dict(), cri_save_path)
        elif (self.act is not None) and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            print("Loaded act:", cwd)
        else:
            print("Act FileNotFound when load_model: {}".format(cwd))
        if (self.cri is not None) and os.path.exists(cri_save_path):
            load_torch_file(self.cri, cri_save_path)
            print("Loaded cri:", cwd)
        else:
            print("Critic FileNotFound when load_model: {}".format(cwd))

    def load_enemy_model(self, cwd):
        """load model files for enemy

        :str cwd: current working directory, we save model file here
        :bool if_save: save model or load model
        """
        act_save_path = '{}/actor.pth'.format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        assert (self.enemy_act is not None) and os.path.exists(
            act_save_path), "Act FileNotFound when load model for enemy"
        load_torch_file(self.enemy_act, act_save_path)
        print("Loaded act:", cwd, " for enemy")


class AgentDiscretePPO(AgentPPO):
    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, if_build_enemy_act=False,
             trainer_id=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        self.act = ActorDiscretePPO(net_dim, state_dim, action_dim).to(self.device)
        self.enemy_act = ActorDiscretePPO(net_dim, state_dim, action_dim).to(self.device) \
            if if_build_enemy_act else None
        self.cri = CriticAdv(net_dim, state_dim).to(self.device)
        self.cri_target = deepcopy(self.cri) if self.cri_target is True else self.cri

        self.criterion = torch.nn.SmoothL1Loss()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)
        self.trainer_id = trainer_id

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = list()
        logging_list = list()
        episode_rewards = list()
        episode_reward = 0
        env.render()
        env.env.simulator.module_UI.text_training_state = "正在采样..."
        states = self.states
        for _ in range(target_step):
            as_int = []
            actions_for_env = [None for _ in range(env.env.simulator.state.robot_num)]
            as_prob = []
            for i in range(env.env.simulator.state.robot_num):
                if i in env.env.trainer_ids:
                    a_int, a_prob = self.select_action(states[i])
                    as_int.append(a_int)
                    as_prob.append(a_prob)
                    actions_for_env[i] = a_int
                elif i in env.env.nn_enemy_ids:
                    actions_for_env[i] = self.select_enemy_action(states[i])
            next_states, reward, done, _ = env.step(actions_for_env)
            env.render()
            if self.act.Multi_Discrete:
                other = (reward * reward_scale, 0.0 if done else gamma, *as_int[0], *np.concatenate(as_prob[0]))
            else:
                other = (reward * reward_scale, 0.0 if done else gamma, as_int[0], *as_prob[0])

            episode_reward += reward
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
            trajectory_list.append((states[0], other))

            states = env.reset() if done else next_states
        self.states = states
        logging_list.append(np.mean(episode_rewards))
        return trajectory_list, logging_list

    def select_enemy_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.enemy_act.get_max_action(states)  # plan to be get_action_a_noise
        return np.array(action)


"""replay.py"""


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_discrete, if_multi_discrete):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        if if_discrete:
            self.action_dim = 1
        elif if_multi_discrete:
            self.action_dim = action_dim.size
            action_dim = sum(action_dim)
        else:
            self.action_dim = action_dim  # for self.sample_all(
        self.tuple = None
        self.np_torch = torch

        other_dim = 1 + 1 + self.action_dim + action_dim
        # other = (reward, mask, action, a_noise) for continuous action
        # other = (reward, mask, a_int, a_prob) for discrete action
        self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
        self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)

    def append_buffer(self, state, other):  # CPU array to CPU array
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, other):  # CPU array to CPU array
        size = len(other)
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def extend_buffer_from_list(self, trajectory_list):
        state_ary = np.array([item[0] for item in trajectory_list], dtype=np.float32)
        other_ary = np.array([item[1] for item in trajectory_list], dtype=np.float32)
        self.extend_buffer(state_ary, other_ary)

    def sample_batch(self, batch_size):
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],  # reward
                r_m_a[:, 1:2],  # mask = 0.0 if done else gamma
                r_m_a[:, 2:],  # action
                self.buf_state[indices],  # state
                self.buf_state[indices + 1])  # next_state

    def sample_all(self):
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],  # reward
                all_other[:, 1],  # mask = 0.0 if done else gamma
                all_other[:, 2:2 + self.action_dim],  # action
                all_other[:, 2 + self.action_dim:],  # action_noise or action_prob
                torch.as_tensor(self.buf_state[:self.now_len], device=self.device))  # state

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer(self):
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False


'''env.py'''


class PreprocessEnv(gym.Wrapper):  # environment wrapper
    def __init__(self, env, if_print=True):
        self.env = gym.make(env) if isinstance(env, str) else env
        super(PreprocessEnv, self).__init__(self.env)

        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.if_multi_discrete, self.target_return) = get_gym_env_info(self.env, if_print)

        self.reset = self.reset_type
        self.step = self.step_type

    def reset_type(self) -> list:
        states = self.env.reset()
        self.reward_dict = self.env.rewards
        return [np.array(state).astype(np.float32) for state in states]

    def step_type(self, actions) -> (list, float, bool, dict):
        states, reward, done, info = self.env.step(actions)
        self.reward_dict = self.env.rewards
        return [np.array(state).astype(np.float32) for state in states], np.array(reward).astype(np.float32), done, info


def get_gym_env_info(env, if_print) -> (str, int, int, int, int, bool, float):
    """get information of a standard OpenAI gym env.

    The DRL algorithm AgentXXX need these env information for building networks and training.
    env_name: the environment name, such as XxxXxx-v0
    state_dim: the dimension of state
    action_dim: the dimension of continuous action; Or the number of discrete action
    action_max: the max action of continuous action; action_max == 1 when it is discrete action space
    if_discrete: Is this env a discrete action space?
    target_return: the target episode return, if agent reach this score, then it pass this game (env).
    max_step: the steps in an episode. (from env.reset to done). It breaks an episode when it reach max_step

    :env: a standard OpenAI gym environment, it has env.reset() and env.step()
    :bool if_print: print the information of environment. Such as env_name, state_dim ...
    """
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    assert isinstance(env, gym.Env)

    env_name = env.unwrapped.spec.id

    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

    target_return = getattr(env, 'target_return', None)
    target_return_default = getattr(env.spec, 'reward_threshold', None)
    if target_return is None:
        target_return = target_return_default
    if target_return is None:
        target_return = 2 ** 16

    max_step = getattr(env, 'max_step', None)
    max_step_default = getattr(env, '_max_episode_steps', None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2 ** 10

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if_multi_discrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif if_multi_discrete:  # make sure it is multi discrete action space
        action_dim = env.action_space.nvec
        action_max = action_dim * 0 + 1
    elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
        assert not any(env.action_space.high + env.action_space.low)
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    print(
        f"\n| env_name:  {env_name}, action space if_discrete: {if_discrete or if_multi_discrete} if_multi_discrete: {if_multi_discrete}"
        f"\n| state_dim: {state_dim:4}, action_dim: {action_dim}, action_max: {action_max}"
        f"\n| max_step:  {max_step:4}, target_return: {target_return}") if if_print else None
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, if_multi_discrete, target_return


def deepcopy_or_rebuild_env(env):
    try:
        env_eval = deepcopy(env)
    except Exception as error:
        print('| deepcopy_or_rebuild_env, error:', error)
        env_eval = PreprocessEnv(env.env_name, if_print=False)
    return env_eval


'''run.py'''


class Arguments:
    def __init__(self, agent=None, env=None, gpu_id=None, if_on_policy=False):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.cwd = time.strftime("%Y-%m-%d_%H-%M-%S",
                                 time.localtime())  # current work directory. cwd is None means set it automatically
        self.enemy_cwd = None
        self.if_train = True
        self.env = env  # the environment for training
        self.env_eval = None  # the environment for evaluating
        self.gpu_id = gpu_id  # choose the GPU for running. gpu_id is None means set it automatically
        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for training (off-policy)'''
        self.learning_rate = 1e-4
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3
        self.train_actor_step = 0
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256

        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9  # the network width
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 4  # collect target_step, then update network
            self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.
        else:
            self.net_dim = 2 ** 8  # the network width
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.target_step = 2 ** 10  # collect target_step, then update network
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.max_memo = 2 ** 17  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.

        '''Arguments for evaluate'''
        self.eval_gap = 2 ** 5  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 2  # evaluation times
        self.eval_times2 = 2 ** 4  # evaluation times if 'eval_reward > max_reward'
        self.random_seed = 0  # initialize random seed in self.init_before_training()

        self.break_step = 2 ** 2  # break training after 'total_step > break_step'
        self.if_remove = False  # remove the cwd folder? (True, False, None:ask me)
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        self.save_interval = 50  # save the model per how many times of evaluation

        '''Arguments for algorithm'''
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.02
        self.lambda_gae_adv = 0.98

    def init_before_training(self, process_id=0):
        # ppo
        if hasattr(self.agent, 'ratio_clip'):
            self.agent.ratio_clip = self.ratio_clip
        if hasattr(self.agent, 'lambda_entropy'):
            self.agent.lambda_entropy = self.lambda_entropy
        if hasattr(self.agent, 'lambda_gae_adv'):
            self.agent.lambda_gae_adv = self.lambda_gae_adv

        if self.agent is None:
            raise RuntimeError('\n| Why agent=None? Assignment args.agent = AgentXXX please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError('\n| Should be agent=AgentXXX() instead of agent=AgentXXX')
        if self.env is None:
            raise RuntimeError('\n| Why env=None? Assignment args.env = XxxEnv() please.')
        if isinstance(self.env, str) or not hasattr(self.env, 'env_name'):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env). It is a Wrapper.')

        '''set None value automatically'''
        if self.gpu_id is None:  # set gpu_id as '0' in default
            self.gpu_id = '0'

        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{self.env.env_name}_{agent_name}'

        if process_id == 0:
            print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')

            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
            if self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print("| Remove history")
            # os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        gpu_id = self.gpu_id[process_id] if isinstance(self.gpu_id, list) else self.gpu_id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def train_and_evaluate(args):
    args.init_before_training()

    '''basic arguments'''
    cwd = os.path.join('./output', args.cwd)
    cwd = os.path.normpath(cwd)
    if args.enemy_cwd:
        enemy_cwd = os.path.join('./output', args.enemy_cwd)
        enemy_cwd = os.path.normpath(enemy_cwd)
        if_build_enemy_act = True
    else:
        enemy_cwd = None
        if_build_enemy_act = False
    env = args.env
    agent = args.agent
    gpu_id = args.gpu_id
    env_eval = args.env_eval

    if_train = args.if_train
    if if_train:
        os.makedirs(cwd)
        '''保存数据'''
        log_dir = Path(os.path.dirname(os.path.abspath(__file__))
                       + "/wandb_logs") / args.env.env_name / 'NoObstacle' / 'ppo'
        os.makedirs(log_dir, exist_ok=True)
        wandb_run = wandb.init(config=args,
                               project='Robomaster',
                               entity='dujinqi',
                               notes='win rate displayed',
                               name='ppo_NVE_2v2_seed=' + str(args.random_seed),
                               group='static enemy',
                               dir=log_dir,
                               job_type="debug_NVE_2v2",
                               reinit=True)
        wandb_run.config.update(env.env.args)
    else:
        wandb_run = None
    '''training arguments'''
    net_dim = args.net_dim
    # max_memo = args.max_memo
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    repeat_times_policy = args.repeat_times_policy
    learning_rate = args.learning_rate
    if_per_or_gae = args.if_per_or_gae
    if_break_early = args.if_allow_break

    gamma = args.gamma
    reward_scale = args.reward_scale
    soft_update_tau = args.soft_update_tau
    # 开始训练actor的step
    train_actor_step = args.train_actor_step
    '''evaluating arguments'''
    show_gap = args.eval_gap
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    save_interval = args.save_interval

    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    if_multi_discrete = env.if_multi_discrete
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)

    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, if_build_enemy_act=if_build_enemy_act)

    buffer_len = target_step + max_step
    buffer = ReplayBuffer(max_len=buffer_len, state_dim=state_dim, action_dim=action_dim,
                          if_discrete=if_discrete, if_multi_discrete=if_multi_discrete)

    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap,
                          save_interval=save_interval)  # build Evaluator

    '''prepare for training'''
    agent.states = env.reset()
    agent.save_load_model(cwd=cwd, if_save=False)  # 读取上一次训练模型
    if if_build_enemy_act:
        agent.load_enemy_model(cwd=enemy_cwd)
    total_step = 0
    '''testing'''
    while not if_train:
        evaluator.evaluate_save(agent.act, agent.cri, enemy_act=agent.enemy_act)
    '''start training'''
    if_train_actor = False if train_actor_step > 0 else True
    while if_train:
        with torch.no_grad():
            trajectory_list, logging_list = agent.explore_env(env, target_step, reward_scale, gamma)
        steps = len(trajectory_list)
        total_step += steps
        if total_step > train_actor_step and not if_train_actor:
            if_train_actor = True
        buffer.extend_buffer_from_list(trajectory_list)
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, repeat_times_policy, soft_update_tau,
                                         if_train_actor)
        logging_tuple = list(logging_tuple)
        logging_tuple += logging_list
        with torch.no_grad():
            if_reach_goal = evaluator.evaluate_save(agent.act, agent.cri, steps, logging_tuple, wandb_run,
                                                    enemy_act=agent.enemy_act)
            if_train = not ((if_break_early and if_reach_goal)
                            or total_step >= break_step
                            or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')
    if if_train:
        wandb_run.finish()


class Evaluator:
    def __init__(self, cwd, agent_id, eval_times1, eval_times2, eval_gap, env, device, save_interval):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.r_max = -np.inf
        self.total_step = 0

        self.cwd = cwd  # constant
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.env = env
        self.target_return = env.target_return

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = -1  # an early time
        self.save_interval = save_interval
        print(f"{'ID':>2} {'Step':>8} {'MaxR':>8} |"
              f"{'avgR':>8} {'stdR':>8} |{'avgS':>5} {'stdS':>4} |"
              f"{'objC':>8} {'etc.':>8}")
        self.epoch = 0

        self.record_controller_id = 0

    def evaluate_save(self, act, cri, steps=0, log_tuple=None, logger=None, enemy_act=None) -> bool:
        if log_tuple is None:
            log_tuple = [0, 0, 0]
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time > self.eval_gap:
            self.eval_time = time.time()
            rewards_steps_list = []
            rewards_dict = {}
            infos_dict = {}
            self.env.render()
            self.env.env.simulator.module_UI.text_training_state = "正在评估..."
            for _ in range(self.eval_times1):
                reward, step, reward_dicts, info_dict = get_episode_return_and_step(self.env, act, self.device, enemy_act)
                reward_dict = reward_dicts[0]
                rewards_steps_list.append((reward, step))
                for key in reward_dict:
                    if key in rewards_dict:
                        rewards_dict[key].append(reward_dict[key])
                    else:
                        rewards_dict[key] = [reward_dict[key]]
                for key in info_dict:
                    if key in infos_dict:
                        infos_dict[key].append(info_dict[key])
                    else:
                        infos_dict[key] = [info_dict[key]]
            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

            if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
                for _ in range(self.eval_times2 - self.eval_times1):
                    reward, step, reward_dicts, info_dict = get_episode_return_and_step(self.env, act, self.device, enemy_act)
                    reward_dict = reward_dicts[0]
                    rewards_steps_list.append((reward, step))
                    for key in reward_dict:
                        rewards_dict[key].append(reward_dict[key])
                    for key in info_dict:
                        infos_dict[key].append(info_dict[key])
                r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
            for key in rewards_dict:
                rewards_dict[key] = np.mean(rewards_dict[key])
            if r_avg > self.r_max:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)
                # logger.save_state({'env': env}, None)
                '''save policy network in *.pth'''
                act_save_path = f'{self.cwd}/actor_best.pth'
                # print('current dir:'+os.path.dirname(os.path.realpath(__file__)))
                # print('act_save_path:'+act_save_path)
                torch.save(act.state_dict(), os.path.dirname(os.path.realpath(__file__)) + '/' + act_save_path)
                act_save_path = f'{self.cwd}/actor_step:' + str(self.total_step) + '_best.pth'
                torch.save(act.state_dict(), os.path.dirname(os.path.realpath(__file__)) + '/' + act_save_path)
                if logger:
                    logger.save(act_save_path)

                print(f"{self.agent_id:<2} {self.total_step:8.2e} {self.r_max:8.2f} |")  # save policy and print
            elif not self.epoch % self.save_interval:
                '''save policy network in *.pth'''
                act_save_path = f'{self.cwd}/actor_step:' + str(self.total_step) + '.pth'
                torch.save(act.state_dict(), os.path.dirname(os.path.realpath(__file__)) + '/' + act_save_path)
                act_save_path = f'{self.cwd}/critic_step:' + str(self.total_step) + '.pth'
                torch.save(cri.state_dict(), os.path.dirname(os.path.realpath(__file__)) + '/' + act_save_path)

            '''save record in logger'''
            train_infos = {'Epoch': self.epoch,
                           str(self.agent_id) + '_MaxR': self.r_max,
                           str(self.agent_id) + '_avgR': log_tuple[3],
                           str(self.agent_id) + '_avgR_eval': r_avg,
                           str(self.agent_id) + '_stdR_eval': r_std,
                           str(self.agent_id) + '_avgS_eval': s_avg,
                           str(self.agent_id) + '_stdS_eval': s_std,
                           str(self.agent_id) + '_objC': log_tuple[0],
                           str(self.agent_id) + '_objA': log_tuple[1],
                           str(self.agent_id) + '_logprob': log_tuple[2]}
            train_infos.update(rewards_dict)
            train_infos.update(infos_dict)
            logger.log(train_infos, step=self.total_step)
            self.epoch += 1
            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f" {'ID':>2} {'Step':>8} {'TargetR':>8} |{'avgR':>8} {'stdR':>8} |"
                      f"  {'UsedTime':>8}  ########\n"
                      f"{self.agent_id:<2} {self.total_step:8.2e} {self.target_return:8.2f} |"
                      f"{r_avg:8.2f} {r_std:8.2f} |"
                      f"  {self.used_time:>8}  ########")

            # plan to
            # if time.time() - self.print_time > self.show_gap:
            print(f" {self.agent_id:<2} {self.total_step:8.2e} {self.r_max:8.2f} |"
                  f"{r_avg:8.2f} {r_std:8.2f} |{s_avg:5.0f} {s_std:4.0f} |"
                  f"{' '.join(f'{n:8.2f}' for n in log_tuple)}")
        else:
            if_reach_goal = False

        return if_reach_goal

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std


def get_episode_return_and_step(env, act, device, enemy_act=None) -> (float, int):
    episode_return = 0.0  # sum of rewards in an episode
    info_dict = None
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete
    if_multi_discrete = env.if_multi_discrete
    state = env.reset()

    for episode_step in range(max_step):
        a_tensor = [None for _ in range(env.env.simulator.state.robot_num)]
        for i in range(env.env.simulator.state.robot_num):
            if i in env.env.trainer_ids:
                s_tensor = torch.as_tensor((state[i],), device=device)
                a_tensor[i] = act(s_tensor)
            elif i in env.env.nn_enemy_ids:
                s_tensor = torch.as_tensor((state[i],), device=device)
                a_tensor[i] = enemy_act(s_tensor)
        # if if_discrete:
        #     a_tensor = a_tensor.argmax(dim=1)
        #     action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        actions = [None for _ in range(env.env.simulator.state.robot_num)]
        if if_multi_discrete:
            for i, a_tensor_ in enumerate(a_tensor):
                if a_tensor_ is not None:
                    n = 0
                    action = []
                    for action_dim_ in env.action_dim:
                        action.append(a_tensor_[:, n:n + action_dim_].argmax(dim=1).detach().cpu().numpy()[0])
                        n += action_dim_
                    actions[i] = action
        state, reward, done, info_dict = env.step(actions)
        env.render()
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step, env.env.rewards_episode, info_dict


'''DEMO'''


def demo_discrete_action():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentDiscretePPO()
    if_train_robomaster = 1
    if if_train_robomaster:
        args.env = PreprocessEnv(env=gym.make('Robomaster-v0'))
        args.agent.trainer_id = args.env.env.trainer_ids[0]  # 表示训练所有trainer中的第一个，其他trainer会一起共享模型
        args.agent.cri_target = False
        args.reward_scale = 2 ** -1
        args.net_dim = 128
        args.gamma = 0.98
        args.batch_size = 256
        args.repeat_times = 16
        args.repeat_times_policy = 5
        args.target_step = 4096
        args.learning_rate = 1e-4
        args.if_per_or_gae = True
        args.if_allow_break = False
        args.break_step = 20000000
        args.gpu_id = sys.argv[-1][-4]
        args.random_seed = 1

        args.eval_gap = 2 ** 6
        args.eval_times1 = 20
        args.eval_times2 = 30

        args.if_train = True
        # args.cwd = '2022-04-27_16-25-48-zero-sum-self-play'
        # args.enemy_cwd = '2022-05-03_16-31-27-PVE'
    '''train and evaluate'''
    train_and_evaluate(args)


if __name__ == '__main__':
    demo_discrete_action()

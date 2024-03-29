import numpy as np
import torch
import numpy.random as rd


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
            self.action_dim_sum = sum(action_dim)
        else:
            self.action_dim = action_dim  # for self.sample_all(
        self.tuple = None
        self.np_torch = torch

        self.other_dim = 1 + 2 + self.action_dim + self.action_dim_sum
        # other = (reward, mask, action, a_noise) for continuous action
        # other = (reward, mask, a_int, a_prob) for discrete action
        self.buf_other = np.empty((max_len, self.other_dim), dtype=np.float32)
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
        return state_ary.shape[0]

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


class MultiAgentMultiEnvsReplayBuffer(ReplayBuffer):
    def __init__(self, max_len, state_dim, action_dim, if_discrete, if_multi_discrete, env, observation_matrix_shape=[1,25,25],
                 state_rnn_dim=128, if_use_cnn=False, if_use_rnn=False):
        super().__init__(max_len, state_dim, action_dim, if_discrete, if_multi_discrete)
        if env is None:
            raise NotImplementedError
        self.env_num = env.env_num
        self.total_trainers_envs = env.get_trainer_ids()
        self.buf_other = [{trainer: np.empty((max_len, self.other_dim), dtype=np.float32) for trainer in trainers}
                          for trainers in self.total_trainers_envs]
        self.buf_state = [{trainer: np.empty((max_len, state_dim), dtype=np.float32) for trainer in trainers}
                          for trainers in self.total_trainers_envs]
        self.buf_state_cnn = [{trainer: np.empty((max_len, *observation_matrix_shape), dtype=np.float32) for trainer in trainers}
                          for trainers in self.total_trainers_envs]
        self.buf_state_rnn = [{trainer: np.empty((max_len, state_rnn_dim), dtype=np.float32) for trainer in trainers}
                          for trainers in self.total_trainers_envs]
        self.tail_idx = [{trainer: 0 for trainer in trainers}
                          for trainers in self.total_trainers_envs]
        self.if_use_cnn = if_use_cnn
        self.if_use_rnn = if_use_rnn

    def extend_buffer_from_list(self, trajectory_list):
        step_num = 0
        states_ary = []
        others_ary = []
        states_2D_ary = []
        states_rnn_ary = []
        for env_id in range(self.env_num):
            states_ary.append({})
            others_ary.append({})
            if self.if_use_cnn:
                states_2D_ary.append({})
            if self.if_use_rnn:
                states_rnn_ary.append({})
            for n in self.total_trainers_envs[env_id]:
                if n in trajectory_list[env_id]:
                    state_ary = np.array([item[0] for item in trajectory_list[env_id][n]], dtype=np.float32)
                    states_ary[-1][n] = state_ary
                    step_num += state_ary.shape[0]
                    other_ary = np.array([item[-1] for item in trajectory_list[env_id][n]], dtype=np.float32)
                    others_ary[-1][n] = other_ary
                    if self.if_use_cnn:
                        states_2D_ary[-1][n] = np.array([item[1] for item in trajectory_list[env_id][n]], dtype=np.float32)
                    if self.if_use_rnn:
                        states_rnn_ary[-1][n] = np.array([item[2] for item in trajectory_list[env_id][n]], dtype=np.float32)
        self.extend_buffer(states_ary, others_ary, states_2D_ary, states_rnn_ary)
        return step_num

    def extend_buffer(self, states, others, states_2D=None, states_rnn=None):  # CPU array to CPU array
        for env_id in range(self.env_num):
            for n in self.total_trainers_envs[env_id]:
                if states[env_id][n].any():
                    state = states[env_id][n]
                    other = others[env_id][n]
                    size = len(other)
                    self.tail_idx[env_id][n] = size
                    self.buf_state[env_id][n][0:size] = state
                    self.buf_other[env_id][n][0:size] = other
                    if self.if_use_cnn:
                        self.buf_state_cnn[env_id][n][0:size] = states_2D[env_id][n]
                    if self.if_use_rnn:
                        self.buf_state_rnn[env_id][n][0:size] = states_rnn[env_id][n]

    def update_now_len(self):
        self.now_len = np.sum([[tail[key] for key in tail] for tail in self.tail_idx])

    def empty_buffer(self):
        self.now_len = 0
        self.tail_idx = [{trainer: 0 for trainer in trainers}
                          for trainers in self.total_trainers_envs]

    def sample_all(self):
        reward = [{} for _ in range(self.env_num)]
        mask = [{} for _ in range(self.env_num)]
        pseudo_mask = [{} for _ in range(self.env_num)]
        action = [{} for _ in range(self.env_num)]
        action_noise = [{} for _ in range(self.env_num)]
        state = [{} for _ in range(self.env_num)]
        state_cnn = [{} for _ in range(self.env_num)]
        state_rnn = [{} for _ in range(self.env_num)]
        for env_id in range(self.env_num):
            for trainer in self.total_trainers_envs[env_id]:
                tail_idx = self.tail_idx[env_id][trainer]
                buf_other = self.buf_other[env_id][trainer]
                buf_state = self.buf_state[env_id][trainer]
                reward[env_id][trainer] = torch.as_tensor(buf_other[0:tail_idx, 0], device=self.device)
                mask[env_id][trainer] = torch.as_tensor(buf_other[0:tail_idx, 1], device=self.device)
                pseudo_mask[env_id][trainer] = torch.as_tensor(buf_other[0:tail_idx, 2], device=self.device)
                action[env_id][trainer] = torch.as_tensor(buf_other[0:tail_idx, 3:3 + self.action_dim], device=self.device)
                action_noise[env_id][trainer] = torch.as_tensor(buf_other[0:tail_idx, 3 + self.action_dim:], device=self.device)
                state[env_id][trainer] = torch.as_tensor(buf_state[0:tail_idx], device=self.device)
                if self.if_use_cnn:
                    state_cnn[env_id][trainer] = torch.as_tensor(self.buf_state_cnn[env_id][trainer][0:tail_idx], device=self.device)
                if self.if_use_rnn:
                    state_rnn[env_id][trainer] = torch.as_tensor(self.buf_state_rnn[env_id][trainer][0:tail_idx],
                                                                 device=self.device)
        return reward, mask, pseudo_mask, action, action_noise, state, state_cnn, state_rnn


class PlugInReplayBuffer(ReplayBuffer):
    def __init__(self, max_len, state_dim, action_dim, if_discrete, if_multi_discrete, env, total_trainers_envs,
                 observation_matrix_shape=[1,25,25], if_use_cnn=False, **kwargs):
        super().__init__(max_len, state_dim, action_dim, if_discrete, if_multi_discrete)
        if env is None:
            raise NotImplementedError
        self.env_num = env.env_num
        self.total_trainers_envs = total_trainers_envs
        self.max_len_per_env = max_len//self.env_num*2
        '''rnn'''
        self.if_use_rnn = True
        self.rnn_hidden_size = 128
        self.LSTM_or_GRU = True
        # kwargs:
        # 是否启用上帝视角critic：use_god_critic
        # 是否使用上帝视角动作预测：use_action_prediction
        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.use_extra_state_for_critic:
            if self.use_action_prediction:
                self.other_dim += (self.agent_num - 1)*sum(self.action_prediction_dim)
        self.buf_other = [{trainer: np.empty((self.max_len_per_env, self.other_dim), dtype=np.float32)
                           for trainer in trainers}
                          for trainers in self.total_trainers_envs]
        self.buf_state = [{trainer: np.empty((self.max_len_per_env, state_dim), dtype=np.float32)
                           for trainer in trainers}
                          for trainers in self.total_trainers_envs]
        if if_use_cnn:
            self.buf_state_cnn = [{trainer: np.empty((self.max_len_per_env, *observation_matrix_shape), dtype=np.float32)
                                                    for trainer in trainers}
                                                  for trainers in self.total_trainers_envs]
        if self.if_use_rnn:
            self.buf_rnn_state = [{trainer: np.empty((self.max_len_per_env, self.rnn_hidden_size), dtype=np.float32)
                                  for trainer in trainers}
                                  for trainers in self.total_trainers_envs]
            if self.LSTM_or_GRU:
                self.buf_LSTM_cell = [{trainer: np.empty((self.max_len_per_env, self.rnn_hidden_size), dtype=np.float32)
                                      for trainer in trainers}
                                      for trainers in self.total_trainers_envs]
        self.tail_idx = [{trainer: 0 for trainer in trainers}
                          for trainers in self.total_trainers_envs]
        self.if_use_cnn = if_use_cnn

    def extend(self, states, others, states_cnn=None, states_rnn=None, env_id=0, agent_id=0):  # CPU array to GPU array
        size = len(states)
        cur_head = self.tail_idx[env_id][agent_id]
        self.tail_idx[env_id][agent_id] += size
        self.next_idx += size
        assert self.tail_idx[env_id][agent_id] < self.max_len_per_env, 'self.tail_idx[env_id][agent_id] error'
        cur_tail = self.tail_idx[env_id][agent_id]
        self.buf_state[env_id][agent_id][cur_head:cur_tail] = states
        self.buf_other[env_id][agent_id][cur_head:cur_tail] = others
        if self.if_use_cnn:
            self.buf_state_cnn[env_id][agent_id][cur_head:cur_tail] = states_cnn
        if self.if_use_rnn:
            self.buf_rnn_state[env_id][agent_id][cur_head:cur_tail] = states_rnn[0]
            if self.LSTM_or_GRU:
                self.buf_LSTM_cell[env_id][agent_id][cur_head:cur_tail] = states_rnn[1]

    def add(self, state, other, state_cnn=None, state_rnn=None, env_id=0, agent_id=0):  # CPU array to GPU array
        cur_head = self.tail_idx[env_id][agent_id]
        self.tail_idx[env_id][agent_id] += 1
        self.next_idx += 1
        assert self.tail_idx[env_id][agent_id] < self.max_len_per_env, 'self.tail_idx[env_id][agent_id] error'
        self.buf_state[env_id][agent_id][cur_head] = state
        self.buf_other[env_id][agent_id][cur_head] = other
        if self.if_use_cnn:
            self.buf_state_cnn[env_id][agent_id][cur_head] = state_cnn
        if self.if_use_rnn:
            self.buf_rnn_state[env_id][agent_id][cur_head] = state_rnn[0]
            if self.LSTM_or_GRU:
                self.buf_LSTM_cell[env_id][agent_id][cur_head] = state_rnn[1]

    def empty_buffer(self):
        self.tail_idx = [{trainer: 0 for trainer in trainers}
                          for trainers in self.total_trainers_envs]
        self.next_idx = 0

    def sample_all(self):
        reward = [{} for _ in range(self.env_num)]
        mask = [{} for _ in range(self.env_num)]
        pseudo_mask = [{} for _ in range(self.env_num)]
        action = [{} for _ in range(self.env_num)]
        action_noise = [{} for _ in range(self.env_num)]
        state = [{} for _ in range(self.env_num)]
        extra_state = [{} for _ in range(self.env_num)]
        state_cnn = [{} for _ in range(self.env_num)]
        state_rnn = [{} for _ in range(self.env_num)]
        for env_id in range(self.env_num):
            for trainer in self.total_trainers_envs[env_id]:
                tail_idx = self.tail_idx[env_id][trainer]
                buf_other = self.buf_other[env_id][trainer]
                buf_state = self.buf_state[env_id][trainer]
                reward[env_id][trainer] = torch.as_tensor(buf_other[0:tail_idx, 0], device=self.device)
                mask[env_id][trainer] = torch.as_tensor(buf_other[0:tail_idx, 1], device=self.device)
                pseudo_mask[env_id][trainer] = torch.as_tensor(buf_other[0:tail_idx, 2], device=self.device)
                action[env_id][trainer] = torch.as_tensor(buf_other[0:tail_idx, 3:3 + self.action_dim], device=self.device)
                action_noise[env_id][trainer] = torch.as_tensor(buf_other[0:tail_idx,
                                    3 + self.action_dim:3 + self.action_dim+self.action_dim_sum], device=self.device)
                if self.use_extra_state_for_critic:
                    extra_state[env_id][trainer] = torch.as_tensor(buf_other[0:tail_idx,
                                                                   3 + self.action_dim+self.action_dim_sum:],
                                                                   device=self.device)
                state[env_id][trainer] = torch.as_tensor(buf_state[0:tail_idx], device=self.device)
                if self.if_use_cnn:
                    state_cnn[env_id][trainer] = torch.as_tensor(self.buf_state_cnn[env_id][trainer][0:tail_idx], device=self.device)
                if self.if_use_rnn:
                    state_rnn[env_id][trainer] = [torch.as_tensor(self.buf_rnn_state[env_id][trainer][0:tail_idx], device=self.device)]
                    if self.LSTM_or_GRU:
                        state_rnn[env_id][trainer].append(torch.as_tensor(self.buf_LSTM_cell[env_id][trainer][0:tail_idx], device=self.device))
        return reward, mask, pseudo_mask, action, action_noise, state, state_cnn, state_rnn, extra_state

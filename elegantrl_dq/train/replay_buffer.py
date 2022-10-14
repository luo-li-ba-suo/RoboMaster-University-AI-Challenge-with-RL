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
            action_dim = sum(action_dim)
        else:
            self.action_dim = action_dim  # for self.sample_all(
        self.tuple = None
        self.np_torch = torch

        self.other_dim = 1 + 2 + self.action_dim + action_dim
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
    def __init__(self, max_len, state_dim, action_dim, if_discrete, if_multi_discrete, env, state_matrix_shape=[1,25,25],
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
        self.buf_state_matrix = [{trainer: np.empty((max_len, state_matrix_shape[0], state_matrix_shape[1], state_matrix_shape[2]), dtype=np.float32) for trainer in trainers}
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
                        self.buf_state_matrix[env_id][n][0:size] = states_2D[env_id][n]
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
        state_2D = [{} for _ in range(self.env_num)]
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
                    state_2D[env_id][trainer] = torch.as_tensor(self.buf_state_matrix[env_id][trainer][0:tail_idx], device=self.device)
                if self.if_use_rnn:
                    state_rnn[env_id][trainer] = torch.as_tensor(self.buf_state_rnn[env_id][trainer][0:tail_idx],
                                                                 device=self.device)
        return reward, mask, pseudo_mask, action, action_noise, state, state_2D, state_rnn

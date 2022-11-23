import os
from copy import deepcopy
import time

from elegantrl_dq.agents.net import *
from elegantrl_dq.agents.model_pool import *
"""agent.py"""


class AgentPPO:
    def __init__(self):
        super().__init__()
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.02
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None

        self.state = None
        self.info_dict = None
        self.device = None
        self.criterion = None
        self.act = self.enemy_act = self.act_optimizer = None
        self.cri = self.cri_optimizer = self.cri_target = None
        self.if_share_network = False

        self.if_use_rnn = False
        self.if_use_cnn = False
        
        self.iteration = 0

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

        state = self.state
        for _ in range(target_step):
            action, noise = self.select_action(state)
            next_state, reward, done, _ = env.step(np.tanh(action))

            other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
            trajectory_list.append((state, other))

            state = env.reset() if done else next_state
        self.state = state
        return trajectory_list

    def adjust_learning_rate(self, optimizer, decay_rate=.99, min_lr=1e-5, decay_start_iteration=200):
        if self.iteration > decay_start_iteration:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(min_lr, param_group['lr'] * decay_rate)

    def update_net(self, buffer, batch_size, repeat_times, repeat_times_policy, soft_update_tau, if_train_actor=True):
        buffer.update_now_len()
        buf_len = buffer.now_len
        buf_state, buf_action, buf_r_sum, buf_logprob, buf_advantage, buf_state_2D, buf_state_rnn, buf_state_extra = self.prepare_buffer(buffer)
        buffer.empty_buffer()

        update_policy_net = int(buf_len / batch_size) * repeat_times_policy  # 策略更新只利用一次数据

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = logprob = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            if self.use_extra_state_for_critic:
                state_extra = buf_state_extra[indices]
            if self.if_use_cnn:
                state_2D = buf_state_2D[indices]
            else:
                state_2D = None
            if self.if_use_rnn:
                state_rnn = [b[indices] for b in buf_state_rnn]
            else:
                state_rnn = None

            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

            if update_policy_net > 0 or self.if_share_network:
                new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action, state_cnn=state_2D,
                                                                        state_rnn=state_rnn)  # it is obj_actor
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = advantage * ratio
                surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
                if if_train_actor and not self.if_share_network:
                    self.optim_update(self.act_optimizer, obj_actor)
                update_policy_net -= 1
            if self.use_extra_state_for_critic:
                state_critic = torch.cat([state, state_extra], dim=-1)
            else:
                state_critic = state
            value = self.cri(state_critic,
                             state_cnn=state_2D if self.if_use_cnn else None,
                             rnn_state=state_rnn if self.if_use_rnn else None).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            if self.if_share_network:
                self.optim_update(self.act_optimizer, obj_actor + obj_critic)
            else:
                self.optim_update(self.cri_optimizer, obj_critic)
                self.soft_update(self.cri_target, self.cri,
                                 soft_update_tau) if self.cri_target is not self.cri else None
        self.iteration += 1
        # # 衰减学习率
        # self.adjust_learning_rate(self.act_optimizer)
        # if not self.if_share_network:
        #     self.adjust_learning_rate(self.cri_optimizer)
        return obj_critic.item(), obj_actor.item(), logprob.mean().item()  # logging_tuple

    def prepare_buffer(self, buffer):
        buf_len = buffer.now_len

        with torch.no_grad():  # compute reverse reward
            reward, mask, action, a_noise, state = buffer.sample_all()

            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            value = torch.cat([self.cri_target(state[i:i + bs]) for i in range(0, state.size(0), bs)], dim=0)
            logprob = self.act.get_old_logprob(action, a_noise)

            pre_state = torch.as_tensor((self.state,), dtype=torch.float32, device=self.device)
            pre_r_sum = self.cri(pre_state).detach()
            r_sum, advantage = self.get_reward_sum(self, buf_len, reward, mask, value, pre_r_sum)
        return state, action, r_sum, logprob, advantage, None, None, None

    @staticmethod
    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value, rest_r_sum) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * rest_r_sum
            rest_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    @staticmethod
    def get_reward_sum_gae(self, buf_len, buf_reward, buf_mask, buf_pseudo_mask, buf_value, next_state_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_advantage = rest_r_sum = next_state_value[-1]  # advantage value of previous step
        index = -2
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * rest_r_sum
            if buf_pseudo_mask[i] > 0:
                buf_r_sum[i] += buf_pseudo_mask[i] * next_state_value[index]
            rest_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + buf_mask[i] * pre_advantage - buf_value[i]  # fix a bug here
            if buf_pseudo_mask[i] > 0:
                buf_advantage[i] += buf_pseudo_mask[i] * next_state_value[index]
                index -= 1
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv
        # buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    @staticmethod
    def get_reward_sum_gae_td_lambda(self, buf_len, buf_reward, buf_mask, buf_pseudo_mask, buf_value, next_state_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_advantage = next_state_value[-1]  # advantage value of previous step
        index = -2
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_advantage
            if buf_pseudo_mask[i] > 0:
                buf_r_sum[i] += buf_pseudo_mask[i] * next_state_value[index]
                index -= 1
            buf_advantage[i] = buf_r_sum[i] - buf_value[i]
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv
        # buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
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
        act_optimizer_save_path = '{}/actor_optimizer.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)
        cri_optimizer_save_path = '{}/critic_optimizer.pth'.format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if self.act is not None:
                torch.save(self.act.state_dict(), act_save_path)
            if self.cri is not None and not self.if_share_network:
                torch.save(self.cri.state_dict(), cri_save_path)
        elif (self.act is not None) and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            if os.path.exists(act_optimizer_save_path):
                load_torch_file(self.act_optimizer, act_optimizer_save_path)
            if self.if_share_network:
                print("Loaded act and critic:", cwd)
            else:
                print("Loaded act:", cwd)
        else:
            print("Act FileNotFound when load_model: {}".format(cwd))
        if not self.if_share_network:
            if (self.cri is not None) and os.path.exists(cri_save_path):
                load_torch_file(self.cri, cri_save_path)
                if os.path.exists(cri_optimizer_save_path):
                    load_torch_file(self.cri_optimizer, cri_optimizer_save_path)
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
            act_save_path), "Enemy Act FileNotFound when load model"
        load_torch_file(self.enemy_act, act_save_path)
        if self.if_share_network:
            print("Loaded enemy  act and critic:", cwd)
        else:
            print("Loaded enemy act:", cwd)


class AgentDiscretePPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.last_state = None
        self.last_alive_trainers = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, if_build_enemy_act=False,
             env=None, self_play=False, enemy_policy_share_memory=False):
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
        if env is None:
            raise NotImplementedError
        self.state = env.reset()

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = list()
        trajectory_list_for_each_agent = [list() for _ in env.env.trainer_ids]
        logging_list = list()
        episode_rewards = list()
        episode_reward = 0
        env.env.display_characters("正在采样...")
        states = self.state
        while True:
            as_int = []
            actions_for_env = [None for _ in range(env.env.simulator.state.robot_num)]
            as_prob = []
            for i in range(env.env.simulator.state.robot_num):
                if i in env.env.trainer_ids:
                    a_int, a_prob = self.select_action(states[i])
                    as_int.append(a_int)
                    as_prob.append(a_prob)
                    actions_for_env[i] = a_int
                elif i in env.env.tester_ids:
                    actions_for_env[i] = self.select_enemy_action(states[i])
            self.last_alive_trainers = env.env.trainer_ids
            next_states, rewards, done, self.info_dict = env.step(actions_for_env)

            for i, n in enumerate(self.last_alive_trainers):
                if n in self.info_dict['robots_being_killed_']:
                    other = (rewards[i] * reward_scale, 0.0, *as_int[i], *np.concatenate(as_prob[i]))
                else:
                    other = (rewards[i] * reward_scale, 0.0 if done else gamma, *as_int[i], *np.concatenate(as_prob[i]))
                trajectory_list_for_each_agent[i].append((states[n], other))
            episode_reward += np.mean(rewards)
            if done:
                states = env.reset()
                episode_rewards.append(episode_reward)
                episode_reward = 0
                for i, _ in enumerate(self.last_alive_trainers):
                    trajectory_list += trajectory_list_for_each_agent[i]
                trajectory_list_for_each_agent = [list() for _ in env.env.trainer_ids]
                if len(trajectory_list) >= target_step:
                    break
            else:
                states = next_states
        self.state = states
        self.last_state = states[env.env.trainer_ids[0]]
        logging_list.append(np.mean(episode_rewards))
        return trajectory_list, logging_list

    def select_enemy_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.enemy_act.get_max_action(states)  # plan to be get_action_a_noise
        return np.array(action)

    def prepare_buffer(self, buffer):
        buf_len = buffer.now_len

        with torch.no_grad():  # compute reverse reward
            reward, mask, action, a_noise, state = buffer.sample_all()

            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            value = torch.cat([self.cri_target(state[i:i + bs]) for i in range(0, state.size(0), bs)], dim=0)
            logprob = self.act.get_old_logprob(action, a_noise)
            # self.states[0]表示第一個智能體的狀態
            pre_state = torch.as_tensor((self.last_state,), dtype=torch.float32, device=self.device)
            pre_r_sum = self.cri(pre_state).detach()
            r_sum, advantage = self.get_reward_sum(self, buf_len, reward, mask, value, pre_r_sum)
        return state, action, r_sum, logprob, advantage, None, None


class MultiEnvDiscretePPO(AgentPPO):
    def __init__(self):
        super().__init__()
        # extra state for critic
        self.stochastic_policy_or_deterministic = True
        self.use_extra_state_for_critic = False
        self.use_action_prediction = False
        self.agent_num = 0
        self.action_prediction_dim = None
        self.actor_obs_dim = 0

        # self play
        self.delta_historySP = None
        self.model_pool_capacity_historySP = None
        self.model_pool = None
        self.self_play_mode = 0
        self.self_play = None
        self.enemy_act_update_interval = 0
        self.enemy_update_steps = 0

        self.enemy_act = None
        self.enemy_stochastic_policy = False
        self.models = None
        self.env_num = None
        self.total_trainers_envs = None
        self.last_states = None
        self.if_complete_episode = False
        self.trajectory_list_rest = None
        self.remove_rest_trajectory = False

        self.state_dim = None
        self.observation_matrix_shape = None

        '''RNN'''
        self.if_use_rnn = True
        self.rnn_hidden_size = 0
        self.LSTM_or_GRU = True
        self.rnn_state_trainers = None
        self.rnn_state_testers = None

        '''Evaluation'''
        self.start_time = time.time()
        self.eval_time = -1  # an early time
        self.save_interval = 0
        self.eval_gap = 0
        self.eval_times = 0
        self.r_max = -float('inf')
        self.cwd = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, max_step=0,
             env=None, if_build_enemy_act=False, enemy_policy_share_memory=False,
             if_use_cnn=False, if_use_conv1D=False,
             if_share_network=True, if_new_proc_eval=False, observation_matrix_shape=[1,25,25],
             **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.self_play_mode == 1 and self.self_play:
            self.model_pool = ModelPool(self.model_pool_capacity_historySP, self.self_play_mode, self.delta_historySP)
        self.state_dim = state_dim
        self.observation_matrix_shape = observation_matrix_shape
        self.max_step = max_step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        self.actor_obs_dim = state_dim
        critic_obs_dim = state_dim
        if self.use_extra_state_for_critic:
            if self.use_action_prediction:
                critic_obs_dim += (self.agent_num - 1) * sum(self.action_prediction_dim)
        if if_share_network:
            self.act = DiscretePPOShareNet(net_dim, critic_obs_dim, action_dim,
                                           actor_obs_dim=self.actor_obs_dim,
                                           if_use_cnn=if_use_cnn,
                                           if_use_conv1D=if_use_conv1D,
                                           state_cnn_channel=observation_matrix_shape[0],
                                           state_seq_len=observation_matrix_shape[-1],
                                           if_use_rnn=self.if_use_rnn, rnn_state_size=self.rnn_hidden_size,
                                           LSTM_or_GRU=self.LSTM_or_GRU).to(self.device)
            self.cri = self.act.critic
        else:
            self.act = MultiAgentActorDiscretePPO(net_dim, self.actor_obs_dim, action_dim, if_use_cnn=if_use_cnn,
                                                  state_cnn_channel=observation_matrix_shape[0],
                                                  if_use_conv1D=if_use_conv1D,
                                                  state_seq_len=observation_matrix_shape[-1]).to(self.device)
            self.cri = CriticAdv(net_dim, critic_obs_dim, if_use_cnn=if_use_cnn,
                                 state_cnn_channel=observation_matrix_shape[0],
                                  if_use_conv1D=if_use_conv1D,
                                  state_seq_len=observation_matrix_shape[-1]).to(self.device)
        if self.self_play or if_build_enemy_act:
            self.enemy_act = deepcopy(self.act)

        self.cri_target = deepcopy(self.cri) if self.cri_target is True else self.cri

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        if not if_share_network:
            self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)
        else:
            self.cri_optimizer = None

        if if_new_proc_eval:
            self.act.share_memory()
            if not if_share_network:
                self.cri.share_memory()
            if self.enemy_act and enemy_policy_share_memory:
                self.enemy_act.share_memory()

            # self.models用于传入新进程里的evaluation
            self.models = {'act': self.act,
                           'enemy_act': self.enemy_act,
                           'cri': self.cri,
                           'act_optimizer': self.act_optimizer,
                           'cri_optimizer': self.cri_optimizer,
                           'net_dim': net_dim,
                           'state_dim': state_dim,
                           'action_dim': action_dim,
                           'if_build_enemy_act': self.self_play or if_build_enemy_act}
        else:
            self.models = {'act': self.act,
                           'enemy_act': self.enemy_act,
                           'act_optimizer': self.act_optimizer,
                           'net_dim': net_dim,
                           'state_dim': state_dim,
                           'action_dim': action_dim,
                           'if_build_enemy_act': self.self_play or if_build_enemy_act}

        self.criterion = torch.nn.SmoothL1Loss()

        self.if_use_cnn = if_use_cnn
        self.if_share_network = if_share_network

        if env is None:
            raise NotImplementedError
        self.env_num = env.env_num
        self.state, self.info_dict = env.reset()
        self.total_trainers_envs = np.array([info['trainers_'] for info in self.info_dict], dtype=object)
        self.total_testers_envs = np.array([info['testers_'] for info in self.info_dict], dtype=object)
        if self.if_use_rnn:
            self.rnn_state_trainers = [{trainer: self.init_rnn_hidden_states() for trainer in trainers}
                                       for trainers in self.total_trainers_envs]
            self.rnn_state_testers = [{tester: self.init_rnn_hidden_states() for tester in testers}
                                       for testers in self.total_testers_envs]

        self.other_dim = 1 + 2 + action_dim.size + sum(action_dim)
        if self.use_extra_state_for_critic:
            if self.use_action_prediction:
                self.other_dim += (self.agent_num - 1)*sum(self.action_prediction_dim)
        if self.if_complete_episode:
            self.trajectory_cache = [{trainer_id: {'vector': np.empty((self.max_step, self.state_dim), dtype=np.float32),
                                                   'others': np.empty((self.max_step, self.other_dim), dtype=np.float32),
                                                   'matrix': np.empty((self.max_step, *self.observation_matrix_shape),  dtype=np.float32)
                                                                       if self.if_use_cnn else None}
                                      for trainer_id in trainers} for trainers in self.total_trainers_envs]
            self.cache_tail_idx = [{trainer_id: 0 for trainer_id in trainers} for trainers in self.total_trainers_envs]
        else:
            self.remove_rest_trajectory = False
        return self.total_trainers_envs

    def init_rnn_hidden_states(self):
        if self.LSTM_or_GRU:
            return [np.zeros(self.rnn_hidden_size), np.zeros(self.rnn_hidden_size)]
        else:
            return [np.zeros(self.rnn_hidden_size)]

    def explore_env(self, env, target_step, reward_scale, gamma, replay_buffer=None):
        if self.remove_rest_trajectory:
            self.trajectory_cache = [{trainer_id: {'vector': np.empty((self.max_step, self.state_dim), dtype=np.float32),
                                                   'others': np.empty((self.max_step, self.other_dim), dtype=np.float32),
                                                   'matrix': np.empty((self.max_step, *self.observation_matrix_shape),  dtype=np.float32)
                                                                       if self.if_use_cnn else None}
                                      for trainer_id in trainers} for trainers in self.total_trainers_envs]
            self.cache_tail_idx = [{trainer_id: 0 for trainer_id in trainers} for trainers in self.total_trainers_envs]
            self.state, self.info_dict = env.reset()
        logging_list = list()
        episode_rewards = list()
        win_rate = list()
        episode_reward = [{trainer_id: 0 for trainer_id in trainers} for trainers in self.total_trainers_envs]
        env.display_characters("正在采样...")
        # states.size: [env_num, trainer_num, state_size]
        states_envs = self.state
        step = 0
        last_trainers_envs = None
        last_pseudo_step = 0
        self.last_states = {'vector': [{trainer_id: [] for trainer_id in last_trainers}
                                          for last_trainers in self.total_trainers_envs],
                            'matrix': None, 'rnn': None}
        if self.if_use_cnn:
            self.last_states['matrix'] = [{trainer_id: [] for trainer_id in last_trainers}
                                          for last_trainers in self.total_trainers_envs]
        if self.if_use_rnn:
            if self.LSTM_or_GRU:
                self.last_states['rnn'] = [{trainer_id: [[],[]] for trainer_id in last_trainers}
                                              for last_trainers in self.total_trainers_envs]
            else:
                self.last_states['rnn'] = [{trainer_id: [[]] for trainer_id in last_trainers}
                                              for last_trainers in self.total_trainers_envs]
        if self.use_extra_state_for_critic:
            self.last_states['extra'] = [{trainer_id: [] for trainer_id in last_trainers}
                                          for last_trainers in self.total_trainers_envs]
        if self.use_action_prediction:
            pseudo_step = [{trainer_id: False for trainer_id in last_trainers}
                                          for last_trainers in self.total_trainers_envs]
        end_states = {'vector': [{trainer_id: None for trainer_id in last_trainers}
                                          for last_trainers in self.total_trainers_envs],
                            'matrix': None}
        if self.if_use_cnn:
            end_states['matrix'] = [{trainer_id: None for trainer_id in last_trainers}
                                          for last_trainers in self.total_trainers_envs]
        if self.if_use_rnn:
            end_states['rnn'] = [{trainer_id: None for trainer_id in last_trainers}
                                          for last_trainers in self.total_trainers_envs]
        if self.self_play:
            if self.self_play_mode == 0:
                if self.enemy_act_update_interval == 0:
                    # naive self play
                    self.enemy_act = self.act
            elif self.self_play_mode == 1:
                self.model_pool.push_model(self.act.state_dict())
                last_step = 0
        real_step = 0
        while step < target_step + last_pseudo_step:
            if self.self_play:
                if self.self_play_mode == 1:
                    if self.enemy_update_steps > self.enemy_act_update_interval:
                        self.enemy_update_steps -= self.enemy_act_update_interval
                        self.enemy_act.load_state_dict(self.model_pool.pull_model())
                    self.enemy_update_steps += step - last_step
                    last_step = step
            # 获取训练者的状态与动作
            last_trainers_envs = np.array([info['trainers_'] for info in self.info_dict], dtype=object)
            last_trainers_envs_size = 0
            for last_trainers in last_trainers_envs:
                last_trainers_envs_size += len(last_trainers)
            states_trainers = {'vector': np.zeros((last_trainers_envs_size, self.state_dim)), 'matrix': None}
            if self.if_use_cnn:
                states_trainers['matrix'] = np.zeros((last_trainers_envs_size, *self.observation_matrix_shape))
            if self.if_use_rnn:
                states_trainers['rnn'] = [np.zeros((last_trainers_envs_size, self.rnn_hidden_size))]
                if self.LSTM_or_GRU:
                    states_trainers['rnn'].append(np.zeros((last_trainers_envs_size, self.rnn_hidden_size)))
            trainer_i = 0
            for env_id in range(env.env_num):
                for trainer_id in last_trainers_envs[env_id]:
                    states_trainers['vector'][trainer_i] = states_envs[env_id][trainer_id][0]
                    if self.if_use_cnn:
                        states_trainers['matrix'][trainer_i] = states_envs[env_id][trainer_id][1]
                    if self.if_use_rnn:
                        states_trainers['rnn'][0][trainer_i] = self.rnn_state_trainers[env_id][trainer_id][0]
                        if self.LSTM_or_GRU:
                            states_trainers['rnn'][1][trainer_i] = self.rnn_state_trainers[env_id][trainer_id][1]
                    trainer_i += 1
            # states_trainers: [num_env*num_trainer, state_dim] or
            #                  [state_num_of_categories, num_env*num_trainer, state_dim]
            trainer_actions, actions_prob, rnn_state = self.select_stochastic_action(states_trainers)
            # 更新rnn state
            if self.if_use_rnn:
                trainer_i = 0
                for env_id in range(self.env_num):
                    for trainer_id in last_trainers_envs[env_id]:
                        self.rnn_state_trainers[env_id][trainer_id][0] = rnn_state[0][trainer_i]
                        if self.LSTM_or_GRU:
                            self.rnn_state_trainers[env_id][trainer_id][1] = rnn_state[1][trainer_i]
                        trainer_i += 1
            # 获取测试者的状态与动作
            last_testers_envs = np.array([info['testers_'] for info in self.info_dict], dtype=object)
            last_testers_envs_size = 0
            for last_testers in last_testers_envs:
                last_testers_envs_size += len(last_testers)
            states_testers = {'vector': np.zeros((last_testers_envs_size, self.state_dim)), 'matrix': None}
            if self.if_use_cnn:
                states_testers['matrix'] = np.zeros((last_testers_envs_size, *self.observation_matrix_shape))
            if self.if_use_rnn:
                states_testers['rnn'] = [np.zeros((last_testers_envs_size, self.rnn_hidden_size))]
                if self.LSTM_or_GRU:
                    states_testers['rnn'].append(np.zeros((last_testers_envs_size, self.rnn_hidden_size)))
            tester_i = 0
            for env_id in range(env.env_num):
                for tester_id in last_testers_envs[env_id]:
                    states_testers['vector'][tester_i] = states_envs[env_id][tester_id][0]
                    if self.if_use_cnn:
                        states_testers['matrix'][tester_i] = states_envs[env_id][tester_id][1]
                    if self.if_use_rnn:
                        states_testers['rnn'][0][tester_i] = self.rnn_state_testers[env_id][tester_id][0]
                        if self.LSTM_or_GRU:
                            states_testers['rnn'][1][tester_i] = self.rnn_state_testers[env_id][tester_id][1]
                    tester_i += 1
            if last_testers_envs.any():
                if self.enemy_stochastic_policy:
                    tester_actions, _, rnn_state = self.select_stochastic_action(states_testers)
                else:
                    tester_actions, rnn_state = self.select_deterministic_action(states_testers)
            else:
                tester_actions = None
            # 更新rnn state
            if self.if_use_rnn:
                tester_i = 0
                for env_id in range(self.env_num):
                    for tester_id in last_testers_envs[env_id]:
                        self.rnn_state_testers[env_id][tester_id][0] = rnn_state[0][tester_i]
                        if self.LSTM_or_GRU:
                            self.rnn_state_testers[env_id][tester_id][1] = rnn_state[1][tester_i]
                        tester_i += 1
            # 标准化动作以传入环境
            trainer_i = tester_i = 0
            actions_for_env = [[None for _ in range(len(states_envs[env_id]))]
                               for env_id in range(env.env_num)]
            for env_id in range(env.env_num):
                for trainer_id in last_trainers_envs[env_id]:
                    actions_for_env[env_id][trainer_id] = trainer_actions[trainer_i]
                    trainer_i += 1
                for tester_id in last_testers_envs[env_id]:
                    actions_for_env[env_id][tester_id] = tester_actions[tester_i]
                    tester_i += 1
            if step < target_step:
                states_envs, rewards, done, self.info_dict = env.step(actions_for_env)
                # print(f'{real_step},', self.info_dict)
            else:
                # 最后这一伪步不允许环境自行开始新的伪步
                states_envs, _, done, self.info_dict = env.step(actions_for_env, pseudo_step_flag=False)
                # print(f'{real_step},', self.info_dict)
            trainer_i = 0
            for env_id in range(env.env_num):
                pseudo_step_cur_env = False
                for i, n in enumerate(last_trainers_envs[env_id]):
                    if self.use_extra_state_for_critic:
                        if pseudo_step[env_id][n]:
                            pseudo_step_cur_env = True
                            self.last_states['extra'][env_id][n].append(
                                self.get_one_hot_other_actions(n, self.info_dict[env_id]['last_actions_']))
                            if self.info_dict[env_id]['pseudo_step_'] == 0:
                                pseudo_step[env_id][n] = False
                if last_pseudo_step > 0:
                    for i, n in enumerate(self.total_trainers_envs[env_id]):
                        if self.use_extra_state_for_critic:
                            # last_pseudo_step > 0代表当前为最后一个伪步数
                            # 当pseudo_step[env_id][n]和last_pseudo_step > 0都满足条件时，会添加两次同一个额外状态
                            pseudo_step_cur_env = True
                            self.last_states['extra'][env_id][n].append(
                                self.get_one_hot_other_actions(n, self.info_dict[env_id]['last_actions_']))
                            # 这一个对应end_states
                            # 那么引出一个问题：如果有机器人提前死亡，last_trainers_envs中没有它，
                            # 那么self.last_states['extra']就会缺少它的信息，导致对不齐
                            # 所以这里遍历的对象为self.total_trainers_envs
                            step += 1
                if pseudo_step_cur_env:
                    continue
                for i, n in enumerate(last_trainers_envs[env_id]):
                    episode_reward[env_id][n] += np.mean(rewards[env_id][i])
                    action_prob = [probs[trainer_i] for probs in actions_prob]
                    if self.use_extra_state_for_critic:
                        if self.use_action_prediction:
                            extra_states = self.get_one_hot_other_actions(n, self.info_dict[env_id]['last_actions_'])
                        else:
                            extra_states = []
                    else:
                        extra_states = []
                    if n in self.info_dict[env_id]['robots_being_killed_'] or done[env_id]:
                        if done[env_id]:
                            win_rate.append(self.info_dict[env_id]['win'])
                        mask = 0.0
                        # 有一种特殊情况：
                        # 一个机器人死亡，另一个机器人坚持到了伪终止
                        # 这时死亡机器人是没有伪步的
                        if self.info_dict[env_id]['pseudo_done'] and n not in self.info_dict[env_id]['robots_being_killed_']:
                            assert done[env_id]
                            mask = gamma
                            self.last_states['vector'][env_id][n].append(states_envs[env_id][n][0])
                            if self.if_use_cnn:
                                self.last_states['matrix'][env_id][n].append(states_envs[env_id][n][1])
                            if self.if_use_rnn:
                                self.last_states['rnn'][env_id][n][0].append(self.rnn_state_trainers[env_id][n][0])
                                if self.LSTM_or_GRU:
                                    self.last_states['rnn'][env_id][n][1].append(self.rnn_state_trainers[env_id][n][1])
                            if self.use_action_prediction:
                                assert self.info_dict[env_id]['pseudo_step_'] == 1, f"'pseudo_step_' not in self.info_dict[{env_id}]"
                                pseudo_step[env_id][n] = True
                        other = (rewards[env_id][i] * reward_scale, 0.0, mask,
                                 *trainer_actions[trainer_i], *np.concatenate(action_prob), *extra_states)
                        episode_rewards.append(episode_reward[env_id][n])
                        episode_reward[env_id][n] = 0
                        end_states['vector'][env_id][n] = states_envs[env_id][n][0]
                        if self.if_use_cnn:
                            end_states['matrix'][env_id][n] = states_envs[env_id][n][1]
                        if self.if_use_rnn:
                            end_states['rnn'][env_id][n] = self.rnn_state_trainers[env_id][n]
                            self.rnn_state_trainers[env_id][n] = self.init_rnn_hidden_states()
                        if self.if_complete_episode:
                            self.trajectory_cache[env_id][n]['vector'][self.cache_tail_idx[env_id][n]] = states_trainers['vector'][trainer_i]
                            self.trajectory_cache[env_id][n]['others'][self.cache_tail_idx[env_id][n]] = other
                            if self.if_use_cnn:
                                self.trajectory_cache[env_id][n]['matrix'][self.cache_tail_idx[env_id][n]] = states_trainers['matrix'][trainer_i]
                                replay_buffer.extend(
                                    self.trajectory_cache[env_id][n]['vector'][:self.cache_tail_idx[env_id][n]+1],
                                    self.trajectory_cache[env_id][n]['others'][:self.cache_tail_idx[env_id][n]+1],
                                    self.trajectory_cache[env_id][n]['matrix'][:self.cache_tail_idx[env_id][n]+1],
                                    env_id=env_id, agent_id=n)
                            else:
                                replay_buffer.extend(
                                    self.trajectory_cache[env_id][n]['vector'][:self.cache_tail_idx[env_id][n]+1],
                                    self.trajectory_cache[env_id][n]['others'][:self.cache_tail_idx[env_id][n]+1],
                                    env_id=env_id, agent_id=n)

                            step += self.cache_tail_idx[env_id][n]+1
                            self.cache_tail_idx[env_id][n] = 0
                    else:
                        other = (rewards[env_id][i] * reward_scale, gamma, 0.0,
                                 *trainer_actions[trainer_i], *np.concatenate(action_prob), *extra_states)
                        if self.if_complete_episode:
                            self.trajectory_cache[env_id][n]['vector'][self.cache_tail_idx[env_id][n]] = states_trainers['vector'][trainer_i]
                            self.trajectory_cache[env_id][n]['others'][self.cache_tail_idx[env_id][n]] = other
                            if self.if_use_cnn:
                                self.trajectory_cache[env_id][n]['matrix'][self.cache_tail_idx[env_id][n]] = states_trainers['matrix'][trainer_i]
                            self.cache_tail_idx[env_id][n] += 1
                            assert self.cache_tail_idx[env_id][n] < self.max_step, "self.cache_tail_idx[env_id][n] error"
                    if not self.if_complete_episode:
                        if not self.if_use_rnn and not self.if_use_cnn:
                            replay_buffer.add(states_trainers['vector'][trainer_i], other,
                                                        env_id=env_id, agent_id=n)
                        elif self.if_use_cnn and not self.if_use_rnn:
                            replay_buffer.add(states_trainers['vector'][trainer_i], other,
                                              states_trainers['matrix'][trainer_i],
                                                        env_id=env_id, agent_id=n)
                        elif self.if_use_cnn and self.if_use_rnn:
                            if self.LSTM_or_GRU:
                                replay_buffer.add(states_trainers['vector'][trainer_i], other,
                                                  states_trainers['matrix'][trainer_i],
                                                  [states_trainers['rnn'][0][trainer_i], states_trainers['rnn'][1][trainer_i]],
                                                  env_id=env_id, agent_id=n)
                            else:
                                replay_buffer.add(states_trainers['vector'][trainer_i], other,
                                                  states_trainers['matrix'][trainer_i],
                                                  [states_trainers['rnn'][0][trainer_i]],
                                                  env_id=env_id, agent_id=n)
                        elif not self.if_use_cnn and self.if_use_rnn:
                            if self.LSTM_or_GRU:
                                replay_buffer.add(states_trainers['vector'][trainer_i], other,
                                                  state_rnn=[states_trainers['rnn'][0][trainer_i], states_trainers['rnn'][1][trainer_i]],
                                                  env_id=env_id, agent_id=n)
                            else:
                                replay_buffer.add(states_trainers['vector'][trainer_i], other,
                                                  state_rnn=[states_trainers['rnn'][0][trainer_i]],
                                                  env_id=env_id, agent_id=n)
                        step += 1
                    trainer_i += 1
                for i, n in enumerate(last_testers_envs[env_id]):
                    if n in self.info_dict[env_id]['robots_being_killed_'] or done[env_id]:
                        if self.if_use_rnn:
                            self.rnn_state_testers[env_id][n] = self.init_rnn_hidden_states()
            if not self.use_action_prediction and step >= target_step:
                real_step = step
                break
            if self.use_action_prediction and step >= target_step and last_pseudo_step == 0:
                for env_id in range(env.env_num):
                    for n in last_trainers_envs[env_id]:
                        if n not in self.info_dict[env_id]['robots_being_killed_']:
                            # 已死亡的智能体已经存好了end_states，所以无须再提取
                            end_states['vector'][env_id][n] = states_envs[env_id][n][0]
                            if self.if_use_cnn:
                                end_states['matrix'][env_id][n] = states_envs[env_id][n][1]
                            if self.if_use_rnn:
                                end_states['rnn'][env_id][n] = self.rnn_state_trainers[env_id][n]
                # 如果还没开始最后一个伪步数（last_pseudo_step == 0）
                # 则给last_pseudo_step赋值，使得该while循环还能再进行一次
                real_step = step
                last_pseudo_step = step - target_step + 1
        # 更新敌方策略
        # if self.self_play and np.mean(win_rate) >= 0.55:
        if self.self_play and self.self_play_mode == 0 and self.enemy_act_update_interval > 0:
            self.update_enemy_policy(real_step)
        if not self.if_complete_episode:
            if not self.use_action_prediction:
                for env_id in range(env.env_num):
                    for n in last_trainers_envs[env_id]:
                        if n not in self.info_dict[env_id]['robots_being_killed_']:
                            # 已死亡的智能体已经存好了end_states，所以无须再提取
                            end_states['vector'][env_id][n] = states_envs[env_id][n][0]
                            if self.if_use_cnn:
                                end_states['matrix'][env_id][n] = states_envs[env_id][n][1]
                            if self.if_use_rnn:
                                end_states['rnn'][env_id][n] = self.rnn_state_trainers[env_id][n]
        self.state = states_envs
        for env_id in range(env.env_num):
            # 将最后一个状态单独存起来
            for n in self.total_trainers_envs[env_id]:
                self.last_states['vector'][env_id][n].append(end_states['vector'][env_id][n])
                if self.if_use_cnn:
                    self.last_states['matrix'][env_id][n].append(end_states['matrix'][env_id][n])
                if self.if_use_rnn:
                    self.last_states['rnn'][env_id][n][0].append(self.rnn_state_trainers[env_id][n][0])
                    if self.LSTM_or_GRU:
                        self.last_states['rnn'][env_id][n][1].append(self.rnn_state_trainers[env_id][n][1])
        if self.use_extra_state_for_critic:
            for env_id in range(env.env_num):
                for trainers in self.total_trainers_envs:
                    for trainer in trainers:
                        assert len(self.last_states['vector'][env_id][trainer]) == len(self.last_states['extra'][env_id][trainer]), f"env_id{env_id} trainer{trainer} {len(self.last_states['vector'][env_id][trainer])} != {len(self.last_states['extra'][env_id][trainer])}"
        # 由代码逻辑可知episode_rewards中不会包含不完整轨迹的回报
        logging_list.append(np.mean(episode_rewards))
        logging_list.append(np.mean(win_rate))
        return logging_list, real_step

    def evaluate(self, env_eval, gamma, if_save=True, steps=0, log_tuple=None, logger=None):
        if log_tuple is None:
            log_tuple = [0, 0, 0, 0, 0]
        if time.time() - self.eval_time > self.eval_gap:
            self.eval_time = time.time()
            infos_dict = {}
            episode_rewards = list()
            episode_returns = list()
            episode_steps = list()
            win_rate = list()
            env_eval.display_characters("正在评估...")

            states_envs, info_dict = env_eval.reset(evaluation=True)
            total_trainers_envs = np.array([info['trainers_'] for info in info_dict], dtype=object)

            episode_reward = [{trainer_id: 0 for trainer_id in trainers} for trainers in total_trainers_envs]
            episode_return = [{trainer_id: 0 for trainer_id in trainers} for trainers in total_trainers_envs]
            episode_step = [{trainer_id: 0 for trainer_id in trainers} for trainers in total_trainers_envs]
            rnn_state_trainers = None
            rnn_state_testers = None
            if self.if_use_rnn:
                rnn_state_trainers = [{trainer: self.init_rnn_hidden_states() for trainer in trainers}
                                           for trainers in self.total_trainers_envs]
                rnn_state_testers = [{tester: self.init_rnn_hidden_states() for tester in testers}
                                          for testers in self.total_testers_envs]
            done_times = 0
            while done_times < self.eval_times:
                # 获取训练者的状态与动作
                last_trainers_envs = np.array([info['trainers_'] for info in info_dict], dtype=object)
                last_trainers_envs_size = 0
                for last_trainers in last_trainers_envs:
                    last_trainers_envs_size += len(last_trainers)
                states_trainers = {'vector': np.zeros((last_trainers_envs_size, self.state_dim)), 'matrix': None}
                if self.if_use_cnn:
                    states_trainers['matrix'] = np.zeros((last_trainers_envs_size, *self.observation_matrix_shape))
                if self.if_use_rnn:
                    states_trainers['rnn'] = [np.zeros((last_trainers_envs_size, self.rnn_hidden_size))]
                    if self.LSTM_or_GRU:
                        states_trainers['rnn'].append(np.zeros((last_trainers_envs_size, self.rnn_hidden_size)))
                trainer_i = 0
                for env_id in range(env_eval.env_num):
                    for trainer_id in last_trainers_envs[env_id]:
                        states_trainers['vector'][trainer_i] = states_envs[env_id][trainer_id][0]
                        if self.if_use_cnn:
                            states_trainers['matrix'][trainer_i] = states_envs[env_id][trainer_id][1]
                        if self.if_use_rnn:
                            states_trainers['rnn'][0][trainer_i] = rnn_state_trainers[env_id][trainer_id][0]
                            if self.LSTM_or_GRU:
                                states_trainers['rnn'][1][trainer_i] = rnn_state_trainers[env_id][trainer_id][1]
                        trainer_i += 1
                # states_trainers: [num_env*num_trainer, state_dim] or
                #                  [state_num_of_categories, num_env*num_trainer, state_dim]
                if self.stochastic_policy_or_deterministic:
                    trainer_actions, _, rnn_state = self.select_stochastic_action(states_trainers)
                else:
                    trainer_actions, rnn_state = self.select_deterministic_action(states_trainers)
                # 更新rnn state
                if self.if_use_rnn:
                    trainer_i = 0
                    for env_id in range(env_eval.env_num):
                        for trainer_id in last_trainers_envs[env_id]:
                            rnn_state_trainers[env_id][trainer_id][0] = rnn_state[0][trainer_i]
                            if self.LSTM_or_GRU:
                                rnn_state_trainers[env_id][trainer_id][1] = rnn_state[1][trainer_i]
                            trainer_i += 1
                # 获取测试者的状态与动作
                last_testers_envs = np.array([info['testers_'] for info in info_dict], dtype=object)
                last_testers_envs_size = 0
                for last_testers in last_testers_envs:
                    last_testers_envs_size += len(last_testers)
                states_testers = {'vector': np.zeros((last_testers_envs_size, self.state_dim))}
                if self.if_use_cnn:
                    states_testers['matrix'] = np.zeros((last_testers_envs_size, *self.observation_matrix_shape))
                if self.if_use_rnn:
                    states_testers['rnn'] = [np.zeros((last_testers_envs_size, self.rnn_hidden_size))]
                    if self.LSTM_or_GRU:
                        states_testers['rnn'].append(np.zeros((last_testers_envs_size, self.rnn_hidden_size)))
                tester_i = 0
                for env_id in range(env_eval.env_num):
                    for tester_id in last_testers_envs[env_id]:
                        states_testers['vector'][tester_i] = states_envs[env_id][tester_id][0]
                        if self.if_use_cnn:
                            states_testers['matrix'][tester_i] = states_envs[env_id][tester_id][1]
                        if self.if_use_rnn:
                            states_testers['rnn'][0][tester_i] = rnn_state_testers[env_id][tester_id][0]
                            if self.LSTM_or_GRU:
                                states_testers['rnn'][1][tester_i] = rnn_state_testers[env_id][tester_id][1]
                        tester_i += 1
                if last_testers_envs.any():
                    if self.enemy_stochastic_policy:
                        tester_actions, _, rnn_state = self.select_stochastic_action(states_testers)
                    else:
                        tester_actions, rnn_state = self.select_deterministic_action(states_testers)
                else:
                    tester_actions = None
                # 更新rnn state
                if self.if_use_rnn:
                    tester_i = 0
                    for env_id in range(env_eval.env_num):
                        for tester_id in last_testers_envs[env_id]:
                            rnn_state_testers[env_id][tester_id][0] = rnn_state[0][tester_i]
                            if self.LSTM_or_GRU:
                                rnn_state_testers[env_id][tester_id][1] = rnn_state[1][tester_i]
                            tester_i += 1
                # 标准化动作以传入环境
                trainer_i = tester_i = 0
                actions_for_env = [[None for _ in range(len(states_envs[env_id]))]
                                   for env_id in range(env_eval.env_num)]
                for env_id in range(env_eval.env_num):
                    for trainer_id in last_trainers_envs[env_id]:
                        actions_for_env[env_id][trainer_id] = trainer_actions[trainer_i]
                        trainer_i += 1
                    for tester_id in last_testers_envs[env_id]:
                        actions_for_env[env_id][tester_id] = tester_actions[tester_i]
                        tester_i += 1
                states_envs, rewards, done, info_dict = env_eval.step(actions_for_env,
                                                                      pseudo_step_flag=False,
                                                                      evaluation=True)
                trainer_i = 0
                for env_id in range(env_eval.env_num):
                    if done[env_id]:
                        for key in info_dict[env_id]:
                            if key[-1] != '_':
                                if 'red_' + key not in infos_dict:
                                    infos_dict['red_' + key] = []
                                    infos_dict['red_' + key].append(info_dict[env_id][key])
                                else:
                                    infos_dict['red_' + key].append(info_dict[env_id][key])
                        done_times += 1
                        win_rate.append(info_dict[env_id]['win'])
                    for i, n in enumerate(last_trainers_envs[env_id]):
                        episode_reward[env_id][n] += np.mean(rewards[env_id][i])
                        episode_return[env_id][n] += gamma ** episode_step[env_id][n] * np.mean(rewards[env_id][i])
                        episode_step[env_id][n] += 1

                        if self.if_use_rnn:
                            rnn_state_trainers[env_id][n] = self.init_rnn_hidden_states()
                        if n in info_dict[env_id]['robots_being_killed_'] or done[env_id]:
                            for key in info_dict[env_id]['reward_record_'][n]:
                                if key[-1] != '_':
                                    if 'reward_' + key not in infos_dict:
                                        infos_dict['reward_' + key] = []
                                        infos_dict['reward_' + key].append(info_dict[env_id]['reward_record_'][n][key])
                                    else:
                                        infos_dict['reward_' + key].append(info_dict[env_id]['reward_record_'][n][key])

                            episode_steps.append(episode_step[env_id][n])
                            episode_step[env_id][n] = 0

                            episode_returns.append(episode_return[env_id][n])
                            episode_return[env_id][n] = 0
                            episode_rewards.append(episode_reward[env_id][n])
                            episode_reward[env_id][n] = 0
                        trainer_i += 1
                    for i, n in enumerate(last_testers_envs[env_id]):
                        if n in info_dict[env_id]['robots_being_killed_'] or done[env_id]:
                            if self.if_use_rnn:
                                rnn_state_testers[env_id][n] = self.init_rnn_hidden_states()
            for key in infos_dict:
                infos_dict[key] = np.mean(infos_dict[key])

            r_avg = np.mean(episode_rewards)
            r_std = np.std(episode_rewards)
            s_avg = np.mean(episode_steps)
            s_std = np.std(episode_steps)

            if r_avg > self.r_max:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)
                if if_save:
                    # logger.save_state({'env': env}, None)
                    '''save policy network in *.pth'''
                    act_save_path = f'{self.cwd}/actor_best.pth'
                    # print('current dir:'+os.path.dirname(os.path.realpath(__file__)))
                    # print('act_save_path:'+act_save_path)
                    torch.save(self.act.state_dict(), act_save_path)
                    act_save_path = f'{self.cwd}/actor_step:' + str(steps) + '_best.pth'
                    torch.save(self.act.state_dict(), act_save_path)
                    if logger:
                        logger.save(act_save_path)

            elif not self.iteration % self.save_interval and if_save:
                '''save policy network in *.pth'''
                act_save_path = f'{self.cwd}/actor_step:' + str(steps) + '.pth'
                torch.save(self.act.state_dict(), act_save_path)
                act_save_path = f'{self.cwd}/critic_step:' + str(steps) + '.pth'
                if not self.if_share_network:
                    torch.save(self.cri.state_dict(), act_save_path)
            if if_save:
                act_save_path = f'{self.cwd}/actor.pth'
                act_optimizer_save_path = f'{self.cwd}/actor_optimizer.pth'
                torch.save(self.act.state_dict(), act_save_path)
                torch.save(self.act_optimizer.state_dict(), act_optimizer_save_path)
                if not self.if_share_network:
                    cri_save_path = f'{self.cwd}/critic.pth'
                    cri_optimizer_save_path = f'{self.cwd}/critic_optimizer.pth'
                    torch.save(self.cri.state_dict(), cri_save_path)
                    torch.save(self.cri_optimizer.state_dict(), cri_optimizer_save_path)
            discounted_returns = np.mean(episode_returns)
            if logger:
                log_tuple[1] = abs(log_tuple[1])
                '''save record in logger'''
                train_infos = {'iteration': self.iteration,
                               'MaxR': self.r_max,
                               'avgR': log_tuple[3],
                               'avgReturn_eval': discounted_returns,
                               'avgR_eval': r_avg,
                               'stdR_eval': r_std,
                               'avgS_eval': s_avg,
                               'stdS_eval': s_std,
                               'objC': log_tuple[0],
                               'objA': log_tuple[1],
                               'log-prob': log_tuple[2],
                               'win_rate_training': log_tuple[4],
                               'actor-learning-rate': self.act_optimizer.param_groups[0]['lr']}
                if self.if_share_network:
                    train_infos['critic-learning-rate'] = self.cri_optimizer.param_groups[0]['lr']
                train_infos.update(infos_dict)
                logger.log(train_infos, step=steps)
            print(f"---Iteration {self.iteration} Steps:{steps:8.2e}".ljust(30, "-"),
                  f"\n| Evaluated {self.eval_times} times".ljust(30, " ") + "|",
                  f"\n| cost time:{time.time() - self.eval_time:8.2f} s".ljust(30, " ") + "|",
                  f"\n| r_avg:{log_tuple[3]:8.2f}".ljust(30, " ") + "|",
                  f"\n| eval_return_avg:{discounted_returns:8.2f}".ljust(30, " ") + "|",
                  f"\n| eval_r_avg:{r_avg:8.2f}".ljust(30, " ") + "|",
                  f"\n| eval_r_max:{self.r_max:8.2f}".ljust(30, " ") + "|",
                  f"\n| eval_r_std:{r_std:8.2f}".ljust(30, " ") + "|",
                  f"\n| average_episode_num:{s_avg:5.0f}".ljust(30, " ") + "|",
                  f"\n| std_episode_num:{s_std:4.0f}".ljust(30, " ") + "|",
                  f"\n| critic loss: {log_tuple[0]:8.4f}".ljust(30, " ") + "|",
                  f"\n| actor loss: {log_tuple[1]:8.4f}".ljust(30, " ") + "|",
                  f"\n| logprob: {log_tuple[2]:8.4f}".ljust(30, " ") + "|",
                  f"\n| red_win_rate:{infos_dict['red_win_rate']:.2f}".ljust(30, " ") + "|",
                  f"\n| red_draw_rate:{infos_dict['red_draw_rate']:.2f}".ljust(30, " ") + "|",
                  f"\n| new red_win_rate:{infos_dict['red_win']:.2f}".ljust(30, " ") + "|",
                  f"\n| new red_fail_rate:{infos_dict['red_fail']:.2f}".ljust(30, " ") + "|",
                  f"\n| win_rate_training: {log_tuple[4]:8.4f}".ljust(30, " ") + "|",
                  "\n---------------------------------".ljust(30, "-"))
        else:
            if logger:
                log_tuple[1] = abs(log_tuple[1])
                '''save record in logger'''
                train_infos = {'iteration': self.iteration,
                               'avgR': log_tuple[3],
                               'objC': log_tuple[0],
                               'objA': log_tuple[1],
                               'log-prob': log_tuple[2],
                               'win_rate_training': log_tuple[4]}
                logger.log(train_infos, step=steps)
            print(f"---Iteration {self.iteration} Steps:{steps:8.2e}".ljust(30, "-"),
                  f"\n| r_avg:{log_tuple[3]:8.2f}".ljust(30, " ") + "|",
                  f"\n| critic loss: {log_tuple[0]:8.4f}".ljust(30, " ") + "|",
                  f"\n| actor loss: {log_tuple[1]:8.4f}".ljust(30, " ") + "|",
                  f"\n| logprob: {log_tuple[2]:8.4f}".ljust(30, " ") + "|",
                  f"\n| win_rate: {log_tuple[4]:8.4f}".ljust(30, " ") + "|",
                  "\n---------------------------------".ljust(30, "-"))

    def get_one_hot_other_actions(self, robot_id, actions):
        action_dim_sum = sum(self.action_prediction_dim)
        one_hot = np.zeros([(self.agent_num-1) * action_dim_sum])
        robot_id_offset = 0
        for i, action in enumerate(actions):
            if i != robot_id:
                offset = 0
                for action_unit in action:
                    if action_unit != -1:
                        one_hot[robot_id_offset + offset + action_unit + 1] = 1
                    offset += self.action_prediction_dim[i]
                robot_id_offset += action_dim_sum
        return one_hot

    def select_stochastic_action(self, state):
        states_1D = state['vector']
        states_2D = None
        states_rnn = None
        states_1D = torch.as_tensor(states_1D[np.newaxis, :], dtype=torch.float32, device=self.device)
        if self.if_use_cnn:
            states_2D = state['matrix']
            if states_2D.ndim < 4:
                states_2D = torch.as_tensor(states_2D[np.newaxis, :], dtype=torch.float32, device=self.device)
            else:
                states_2D = torch.as_tensor(states_2D, dtype=torch.float32, device=self.device)
        if self.if_use_rnn:
            states_rnn = state['rnn']
            states_rnn = [torch.as_tensor(state_rnn, dtype=torch.float32, device=self.device)
                          for state_rnn in states_rnn]
        if states_1D.dim() == 2:
            action_dim = [-1]
        elif states_1D.dim() == 3:
            states_1D = states_1D.squeeze(0)
            action_dim = [states_1D.shape[0], -1]
        else:
            raise NotImplementedError
        actions, noises, rnn_state = self.act.get_stochastic_action(states_1D, states_2D, states_rnn)
        actions = torch.cat([action.unsqueeze(0) for action in actions]).T.view(*action_dim).detach().cpu().numpy()
        noises = [noise.view(*action_dim).detach().cpu().numpy() for noise in noises]
        if self.if_use_rnn:
            rnn_state = [s.detach().cpu().numpy() for s in rnn_state]
        return actions, noises, rnn_state

    def select_deterministic_action(self, state):
        states_1D = state['vector']
        states_2D = None
        states_rnn = None
        states_1D = torch.as_tensor(states_1D[np.newaxis, :], dtype=torch.float32, device=self.device)
        if self.if_use_cnn:
            states_2D = state['matrix']
            if states_2D.ndim < 4:
                states_2D = torch.as_tensor(states_2D[np.newaxis, :], dtype=torch.float32, device=self.device)
            else:
                states_2D = torch.as_tensor(states_2D, dtype=torch.float32, device=self.device)
        if self.if_use_rnn:
            states_rnn = state['rnn']
            states_rnn = (torch.as_tensor(state_rnn, dtype=torch.float32, device=self.device)
                          for state_rnn in states_rnn)
        if states_1D.dim() == 2:
            action_dim = [-1]
        elif states_1D.dim() == 3:
            states_1D = states_1D.squeeze(0)
            action_dim = [states_1D.shape[0], -1]
        else:
            raise NotImplementedError
        actions, rnn_state = self.enemy_act.get_deterministic_action(states_1D, states_2D, states_rnn)
        actions = torch.cat([action.unsqueeze(0) for action in actions]).T.view(*action_dim).detach().cpu().numpy()
        if self.if_use_rnn:
            rnn_state = [s.detach().cpu().numpy() for s in rnn_state]
        return actions, rnn_state

    def prepare_buffer(self, buffer):
        with torch.no_grad():  # compute reverse reward
            states = []
            extra_states = []
            states_2D = []
            states_rnn = [[], []] if self.LSTM_or_GRU else [[]]
            actions = []
            r_sums = []
            logprobs = []
            advantages = []
            reward_samples, mask_samples, pseudo_mask_samples, action_samples, a_noise_samples, state_samples, state_2D_samples, state_rnn_samples, extra_state_samples = buffer.sample_all()
            for env_id in range(self.env_num):
                for trainer in self.total_trainers_envs[env_id]:
                    if trainer in state_samples[env_id]:
                        data_len = len(state_samples[env_id][trainer])
                        if self.use_extra_state_for_critic:
                            value = self.cri_target(torch.cat((state_samples[env_id][trainer],
                                                               extra_state_samples[env_id][trainer]), dim=-1),
                                            state_cnn=state_2D_samples[env_id][trainer] if self.if_use_cnn else None,
                                            rnn_state=state_rnn_samples[env_id][trainer] if self.if_use_rnn else None)
                        else:
                            value = self.cri_target(state_samples[env_id][trainer],
                                                    state_cnn=state_2D_samples[env_id][trainer] if self.if_use_cnn else None,
                                                    rnn_state=state_rnn_samples[env_id][trainer] if self.if_use_rnn else None)
                        logprob = self.act.get_old_logprob(action_samples[env_id][trainer],
                                                           a_noise_samples[env_id][trainer])
                        last_state = torch.as_tensor(np.array(self.last_states['vector'][env_id][trainer]),
                                                     dtype=torch.float32, device=self.device)
                        if self.use_extra_state_for_critic:
                            last_extra_state = torch.as_tensor(np.array(self.last_states['extra'][env_id][trainer]),
                                                                dtype=torch.float32, device=self.device)
                            last_state = torch.cat([last_state, last_extra_state], dim=-1)
                        last_state_2D = None
                        last_rnn_state = None
                        if self.if_use_cnn:
                            last_state_2D = torch.as_tensor(np.array(self.last_states['matrix'][env_id][trainer]),
                                                            dtype=torch.float32, device=self.device)
                        if self.if_use_rnn:
                            last_rnn_state = [torch.as_tensor(rnn_state, dtype=torch.float32, device=self.device)
                                              for rnn_state in np.array(self.last_states['rnn'][env_id][trainer])]
                        rest_r_sum = self.cri_target(last_state,
                                                     last_state_2D if self.if_use_cnn else None,
                                                     last_rnn_state if self.if_use_rnn else None).detach()
                        value = value.squeeze(-1)
                        rest_r_sum = rest_r_sum.squeeze(-1)
                        r_sum, advantage = self.get_reward_sum(self, data_len, reward_samples[env_id][trainer],
                                                               mask_samples[env_id][trainer],
                                                               pseudo_mask_samples[env_id][trainer], value, rest_r_sum)
                        states.append(state_samples[env_id][trainer])
                        if self.use_extra_state_for_critic:
                            extra_states.append(extra_state_samples[env_id][trainer])
                        if self.if_use_cnn:
                            states_2D.append(state_2D_samples[env_id][trainer])
                        if self.if_use_rnn:
                            states_rnn[0].append(state_rnn_samples[env_id][trainer][0])
                            if self.LSTM_or_GRU:
                                states_rnn[1].append(state_rnn_samples[env_id][trainer][1])
                        actions.append(action_samples[env_id][trainer])
                        r_sums.append(r_sum)
                        logprobs.append(logprob)
                        advantages.append(advantage)
            states = torch.cat(states)
            if self.use_extra_state_for_critic:
                extra_states = torch.cat(extra_states)
            if self.if_use_cnn:
                states_2D = torch.cat(states_2D)
            if self.if_use_rnn:
                states_rnn = [torch.cat(state_rnn) for state_rnn in states_rnn]
            actions = torch.cat(actions)
            r_sums = torch.cat(r_sums)
            logprobs = torch.cat(logprobs)
            advantages = torch.cat(advantages)
        return states, actions, r_sums, logprobs, advantages, states_2D, states_rnn, extra_states

    def update_enemy_policy(self, step):
        if self.enemy_update_steps > self.enemy_act_update_interval:
            self.enemy_update_steps = 0
            self.enemy_act.load_state_dict(self.act.state_dict())
        self.enemy_update_steps += step

    def init_hidden_states(self, n):
        return np.zeros((n, self.rnn_dim))

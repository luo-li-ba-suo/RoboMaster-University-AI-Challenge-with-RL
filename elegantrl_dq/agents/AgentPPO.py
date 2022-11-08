import os
from copy import deepcopy

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
        
        self.iteration_num = 0

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
        if self.iteration_num > decay_start_iteration:
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
                state_rnn = buf_state_rnn[indices]
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
            if self.if_use_cnn:
                value = self.cri(state_critic, state_2D).squeeze(1)
            else:
                value = self.cri(state_critic).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            if self.if_share_network:
                self.optim_update(self.act_optimizer, obj_actor + obj_critic)
            else:
                self.optim_update(self.cri_optimizer, obj_critic)
                self.soft_update(self.cri_target, self.cri,
                                 soft_update_tau) if self.cri_target is not self.cri else None
        self.iteration_num += 1
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
        # self play
        self.delta_historySP = None
        self.model_pool_capacity_historySP = None
        self.model_pool = None
        self.self_play_mode = 0
        self.self_play = None
        self.enemy_act_update_interval = 0
        self.enemy_update_steps = 0

        self.enemy_act = None
        self.models = None
        self.env_num = None
        self.total_trainers_envs = None
        self.last_states = None
        self.if_complete_episode = False
        self.trajectory_list_rest = None
        self.remove_rest_trajectory = False

        self.state_dim = None
        self.observation_matrix_shape = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, max_step=0,
             env=None, if_build_enemy_act=False, enemy_policy_share_memory=False,
             if_use_cnn=False, if_use_conv1D=False,
             if_use_rnn=False, if_share_network=True, if_new_proc_eval=False, observation_matrix_shape=[1,25,25],
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
        if if_share_network:
            self.act = DiscretePPOShareNet(net_dim, state_dim, action_dim,
                                           if_use_cnn=if_use_cnn,
                                           if_use_conv1D=if_use_conv1D,
                                           if_use_rnn=if_use_rnn,
                                           state_seq_len=observation_matrix_shape[-1]).to(self.device)
            self.cri = self.act.critic
        else:
            self.act = MultiAgentActorDiscretePPO(net_dim, state_dim, action_dim, if_use_cnn=if_use_cnn,
                                                  state_cnn_channel=observation_matrix_shape[0],
                                                  if_use_rnn=if_use_rnn,
                                                  if_use_conv1D=if_use_conv1D,
                                                  state_seq_len=observation_matrix_shape[-1]).to(self.device)
            input_dim = state_dim
            if self.use_extra_state_for_critic:
                if self.use_action_prediction:
                    input_dim += (self.agent_num - 1) * sum(self.action_prediction_dim)
            self.cri = CriticAdv(net_dim, input_dim, if_use_cnn=if_use_cnn,
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
                           'if_build_enemy_act': self_play or if_build_enemy_act}
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
        self.if_use_rnn = if_use_rnn
        self.if_share_network = if_share_network
        self.rnn_dim = None
        self.rnn_state = np.nan

        if env is None:
            raise NotImplementedError
        self.env_num = env.env_num
        self.state, self.info_dict = env.reset()
        self.total_trainers_envs = np.array([info['trainers_'] for info in self.info_dict], dtype=object)
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
                            'matrix': None}
        if self.if_use_cnn:
            self.last_states['matrix'] = [{trainer_id: [] for trainer_id in last_trainers}
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
            trainer_i = 0
            for env_id in range(env.env_num):
                for trainer_id in last_trainers_envs[env_id]:
                    states_trainers['vector'][trainer_i] = states_envs[env_id][trainer_id][0]
                    if self.if_use_cnn:
                        states_trainers['matrix'][trainer_i] = states_envs[env_id][trainer_id][1]
                    trainer_i += 1
            # states_trainers: [num_env*num_trainer, state_dim] or
            #                  [state_num_of_categories, num_env*num_trainer, state_dim]
            trainer_actions, actions_prob = self.select_stochastic_action(states_trainers)

            # 获取测试者的状态与动作
            last_testers_envs = np.array([info['testers_'] for info in self.info_dict], dtype=object)
            last_testers_envs_size = 0
            for last_testers in last_testers_envs:
                last_testers_envs_size += len(last_testers)
            states_testers = {'vector': np.zeros((last_testers_envs_size, self.state_dim)), 'matrix': None}
            if self.if_use_cnn:
                states_testers['matrix'] = np.zeros((last_testers_envs_size, *self.observation_matrix_shape))
            tester_i = 0
            for env_id in range(env.env_num):
                for tester_id in last_testers_envs[env_id]:
                    states_testers['vector'][tester_i] = states_envs[env_id][tester_id][0]
                    if self.if_use_cnn:
                        states_testers['matrix'][tester_i] = states_envs[env_id][tester_id][1]
                    tester_i += 1
            if last_testers_envs.any():
                tester_actions = self.select_deterministic_action(states_testers)
            else:
                tester_actions = None

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
                _, _, _, self.info_dict = env.step(actions_for_env, pseudo_step_flag=False)
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
                        # last_pseudo_step > 0代表当前为最后一个伪步数
                        # 当pseudo_step[env_id][n]和last_pseudo_step > 0都满足条件时，会添加两次同一个额外状态
                        if last_pseudo_step > 0:
                            pseudo_step_cur_env = True
                            self.last_states['extra'][env_id][n].append(
                                self.get_one_hot_other_actions(n, self.info_dict[env_id]['last_actions_']))
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
                        else:
                            raise NotImplementedError
                        step += 1
                    trainer_i += 1
            if not self.use_action_prediction and step >= target_step:
                real_step = step
                break
            if self.use_action_prediction and step >= target_step and last_pseudo_step == 0:
                for env_id in range(env.env_num):
                    for n in last_trainers_envs[env_id]:
                        # 已死亡的智能体已经存好了end_states，所以无须再提取
                        end_states['vector'][env_id][n] = states_envs[env_id][n][0]
                        if self.if_use_cnn:
                            end_states['matrix'][env_id][n] = states_envs[env_id][n][1]
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
                        # 已死亡的智能体已经存好了end_states，所以无须再提取
                        end_states['vector'][env_id][n] = states_envs[env_id][n][0]
                        if self.if_use_cnn:
                            end_states['matrix'][env_id][n] = states_envs[env_id][n][1]
        self.state = states_envs
        for env_id in range(env.env_num):
            # 将最后一个状态单独存起来
            for n in end_states['vector'][env_id]:
                self.last_states['vector'][env_id][n].append(end_states['vector'][env_id][n])
                if self.if_use_cnn:
                    self.last_states['matrix'][env_id][n].append(end_states['matrix'][env_id][n])
        # 由代码逻辑可知episode_rewards中不会包含不完整轨迹的回报
        logging_list.append(np.mean(episode_rewards))
        logging_list.append(np.mean(win_rate))
        return logging_list, real_step

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
        states_2D = state['matrix']
        states_rnn = None
        states_1D = torch.as_tensor(states_1D[np.newaxis, :], dtype=torch.float32, device=self.device)
        if self.if_use_cnn:
            if states_2D.ndim < 4:
                states_2D = torch.as_tensor(states_2D[np.newaxis, :], dtype=torch.float32, device=self.device)
            else:
                states_2D = torch.as_tensor(states_2D, dtype=torch.float32, device=self.device)
        if self.if_use_rnn:
            states_rnn = torch.as_tensor((states_rnn,), dtype=torch.float32, device=self.device)
        if states_1D.dim() == 2:
            action_dim = [-1]
        elif states_1D.dim() == 3:
            states_1D = states_1D.squeeze(0)
            action_dim = [states_1D.shape[0], -1]
        else:
            raise NotImplementedError
        actions, noises = self.act.get_stochastic_action(states_1D, states_2D, states_rnn)
        actions = torch.cat([action.unsqueeze(0) for action in actions]).T.view(*action_dim).detach().cpu().numpy()
        noises = [noise.view(*action_dim).detach().cpu().numpy() for noise in noises]
        return actions, noises

    def select_deterministic_action(self, state):
        states_1D = state['vector']
        states_2D = state['matrix']
        states_rnn = None
        states_1D = torch.as_tensor(states_1D[np.newaxis, :], dtype=torch.float32, device=self.device)
        if self.if_use_cnn:
            if states_2D.ndim < 4:
                states_2D = torch.as_tensor(states_2D[np.newaxis, :], dtype=torch.float32, device=self.device)
            else:
                states_2D = torch.as_tensor(states_2D, dtype=torch.float32, device=self.device)
        if self.if_use_rnn:
            states_rnn = torch.as_tensor((states_rnn,), dtype=torch.float32, device=self.device)
        if states_1D.dim() == 2:
            action_dim = [-1]
        elif states_1D.dim() == 3:
            states_1D = states_1D.squeeze(0)
            action_dim = [states_1D.shape[0], -1]
        else:
            raise NotImplementedError
        actions = self.enemy_act.get_deterministic_action(states_1D, states_2D, states_rnn)
        actions = torch.cat([action.unsqueeze(0) for action in actions]).T.view(*action_dim).detach().cpu().numpy()
        return actions

    def prepare_buffer(self, buffer):
        with torch.no_grad():  # compute reverse reward
            states = []
            extra_states = []
            states_2D = []
            states_rnn = []
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
                            if self.if_use_cnn:
                                value = self.cri_target(torch.cat((state_samples[env_id][trainer],
                                                                   extra_state_samples[env_id][trainer]), dim=-1),
                                                        state_2D_samples[env_id][trainer])
                            else:
                                value = self.cri_target(torch.cat((state_samples[env_id][trainer],
                                                                   extra_state_samples[env_id][trainer]), dim=-1))
                        else:
                            if self.if_use_cnn:
                                value = self.cri_target(state_samples[env_id][trainer],
                                                        state_2D_samples[env_id][trainer])
                            else:
                                value = self.cri_target(state_samples[env_id][trainer])
                        logprob = self.act.get_old_logprob(action_samples[env_id][trainer],
                                                           a_noise_samples[env_id][trainer])
                        last_state = torch.as_tensor(np.array(self.last_states['vector'][env_id][trainer]),
                                                     dtype=torch.float32, device=self.device)
                        if self.use_extra_state_for_critic:
                            last_extra_state = torch.as_tensor(np.array(self.last_states['extra'][env_id][trainer]),
                                                                dtype=torch.float32, device=self.device)
                            last_state = torch.cat([last_state, last_extra_state], dim=-1)
                        if not self.if_use_cnn:
                            rest_r_sum = self.cri_target(last_state).detach()
                        else:
                            last_state_2D = torch.as_tensor(np.array(self.last_states['matrix'][env_id][trainer]),
                                                            dtype=torch.float32, device=self.device)
                            rest_r_sum = self.cri_target(last_state, last_state_2D).detach()
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
                            states_rnn.append(state_rnn_samples[env_id][trainer])
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
                states_rnn = torch.cat(states_rnn)
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

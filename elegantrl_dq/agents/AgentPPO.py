import os
from copy import deepcopy

import torch

from elegantrl_dq.agents.net import *

"""agent.py"""


class AgentPPO:
    def __init__(self):
        super().__init__()
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.02
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None

        self.state = None
        self.device = None
        self.criterion = None
        self.act = self.enemy_act = self.act_optimizer = None
        self.cri = self.cri_optimizer = self.cri_target = None

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

            pre_state = torch.as_tensor((self.state,), dtype=torch.float32, device=self.device)
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
    def __init__(self):
        super().__init__()
        self.last_state = None
        self.last_alive_trainers = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, if_build_enemy_act=False,
             env=None):
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
                elif i in env.env.nn_enemy_ids:
                    actions_for_env[i] = self.select_enemy_action(states[i])
            self.last_alive_trainers = env.env.trainer_ids
            next_states, rewards, done, info_dict = env.step(actions_for_env)

            for i, n in enumerate(self.last_alive_trainers):
                if n in info_dict['robots_being_killed_']:
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
        return state, action, r_sum, logprob, advantage


class MultiEnvDiscretePPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.models = None
        self.env_num = None
        self.total_trainers_envs = None
        self.last_states = None
        self.if_complete_episode = True
        self.trajectory_list_rest = None
        self.remove_rest_trajectory = True

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False,
             env=None, if_build_enemy_act=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        self.act = MultiAgentActorDiscretePPO(net_dim, state_dim, action_dim).to(self.device)
        self.enemy_act = MultiAgentActorDiscretePPO(net_dim, state_dim, action_dim).to(self.device) \
            if if_build_enemy_act else None
        self.cri = CriticAdv(net_dim, state_dim).to(self.device)
        self.cri_target = deepcopy(self.cri) if self.cri_target is True else self.cri
        self.act.share_memory()
        if self.enemy_act:
            self.enemy_act.share_memory()
        self.cri.share_memory()
        # self.models用于传入新进程里的evaluation
        self.models = {'act': self.act,
                       'enemy_act': self.enemy_act,
                       'cri': self.cri,
                       'net_dim': net_dim,
                       'state_dim': state_dim,
                       'action_dim': action_dim,
                       'if_build_enemy_act': if_build_enemy_act}

        self.criterion = torch.nn.SmoothL1Loss()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)

        if env is None:
            raise NotImplementedError
        self.env_num = env.env_num
        self.state = env.reset()
        self.total_trainers_envs = env.get_trainer_ids()
        # 给last_state填充初始状态，这样可以避免循环结束时缺乏某些死亡智能体的最终状态;尽管他们的最后状态价值为0
        self.last_states = [{trainer_id: self.state[env_num][trainer_id]
                             for trainer_id in last_trainers}
                            for env_num, last_trainers in enumerate(self.total_trainers_envs)]
        if self.if_complete_episode:
            self.trajectory_list_rest = [{trainer_id: [] for trainer_id in trainers} for trainers in
                                         self.total_trainers_envs]

    def explore_env(self, env, target_step, reward_scale, gamma):
        if self.remove_rest_trajectory:
            self.trajectory_list_rest = [{trainer_id: [] for trainer_id in trainers} for trainers in
                                         self.total_trainers_envs]
            self.state = env.reset()
        trajectory_list = [{trainer_id: [] for trainer_id in trainers} for trainers in self.total_trainers_envs]
        logging_list = list()
        episode_rewards = list()
        episode_reward = [{trainer_id: 0 for trainer_id in trainers} for trainers in self.total_trainers_envs]
        env.display_characters("正在采样...")
        # states.size: [env_num, trainer_num, state_size]
        states_envs = self.state
        step = 0
        while step < target_step:
            actions_for_env = [[None for _ in range(len(states_envs[env_id]))]
                               for env_id in range(env.env_num)]
            last_trainers_envs = np.array(env.get_trainer_ids())
            last_testers_envs = np.array(env.get_tester_ids())
            # states_concatenate.size: [trainer_num*env_num, state_size]
            states_concatenate = []
            for env_id in range(env.env_num):
                for trainer_id in last_trainers_envs[env_id]:
                    states_concatenate.append(states_envs[env_id][trainer_id])
            states_concatenate = np.array(states_concatenate)
            stochastic_num = sum([len(l) for l in last_trainers_envs])
            deterministic_num = sum([len(l) for l in last_testers_envs])
            trainer_actions, actions_prob, tester_actions = self.select_action(states_concatenate,
                                                                               stochastic_num,
                                                                               deterministic_num)
            trainer_i = tester_i = 0
            for env_id in range(env.env_num):
                for trainer_id in last_trainers_envs[env_id]:
                    actions_for_env[env_id][trainer_id] = trainer_actions[trainer_i]
                    trainer_i += 1
                for tester_id in last_testers_envs[env_id]:
                    actions_for_env[env_id][tester_id] = tester_actions[tester_i]
                    tester_i += 1
            states_envs, rewards, done, info_dict = env.step(actions_for_env)
            trainer_i = 0
            for env_id in range(env.env_num):
                for i, n in enumerate(last_trainers_envs[env_id]):
                    episode_reward[env_id][n] += np.mean(rewards[env_id][i])
                    action_prob = [probs[trainer_i] for probs in actions_prob]
                    if n in info_dict[env_id]['robots_being_killed_'] or done[env_id]:
                        # if n in info_dict[env_id]['robots_being_killed_']:
                        #     # 这一步实际上没用，因为最后一个状态的价值为0;
                        #     self.last_states[env_id][n] = states[env_id][n]
                        other = (rewards[env_id][i] * reward_scale, 0.0,
                                 *trainer_actions[trainer_i], *np.concatenate(action_prob))
                        episode_rewards.append(episode_reward[env_id][n])
                        episode_reward[env_id][n] = 0
                        if self.if_complete_episode:
                            self.trajectory_list_rest[env_id][n].append((states_concatenate[trainer_i], other))
                            trajectory_list[env_id][n] += self.trajectory_list_rest[env_id][n]
                            step += len(self.trajectory_list_rest[env_id][n])
                            self.trajectory_list_rest[env_id][n] = []
                    else:
                        other = (rewards[env_id][i] * reward_scale, gamma,
                                 *trainer_actions[trainer_i], *np.concatenate(action_prob))
                        if self.if_complete_episode:
                            self.trajectory_list_rest[env_id][n].append((states_concatenate[trainer_i], other))
                    if not self.if_complete_episode:
                        trajectory_list[env_id][n].append((states_concatenate[trainer_i], other))
                        step += 1

                    trainer_i += 1

        self.state = states_envs
        for env_id in range(env.env_num):
            # 在last_trainers_envs中可能缺失了中途死亡的智能体
            for n in last_trainers_envs[env_id]:
                if states_envs[env_id][n].shape:  # TODO:为什么会出现bug？
                    self.last_states[env_id][n] = states_envs[env_id][n]
        # 由代码逻辑可知episode_rewards中不会包含不完整轨迹的回报
        logging_list.append(np.mean(episode_rewards))
        return trajectory_list, logging_list

    def select_action(self, state, stochastic=0, deterministic=0):
        if state.ndim == 1:
            action_dim = [-1]
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        elif state.ndim > 1:
            action_dim = list(state.shape[:-1])
            action_dim.append(-1)
            states = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            states.view(-1, states.shape[-1])
        else:
            raise NotImplementedError
        actions, noises, deterministic_actions = self.act.get_action(states, stochastic,
                                                                     deterministic)  # plan to be get_action_a_noise

        if isinstance(actions, list):  # 如果是Multi-Discrete
            if actions is None:
                raise NotImplementedError
            actions = torch.cat([action.unsqueeze(0) for action in actions]).T.view(*action_dim).detach().cpu().numpy()
            noises = [noise.view(*action_dim).detach().cpu().numpy() for noise in noises]
            if deterministic_actions is not None:
                deterministic_actions = np.array([action.view(*action_dim).detach().cpu().numpy()
                                                  for action in deterministic_actions])
            return actions, noises, deterministic_actions
        else:
            raise NotImplementedError

    def prepare_buffer(self, buffer):
        with torch.no_grad():  # compute reverse reward
            states = []
            actions = []
            r_sums = []
            logprobs = []
            advantages = []
            reward_samples, mask_samples, action_samples, a_noise_samples, state_samples = buffer.sample_all()
            for env_id in range(self.env_num):
                for trainer in self.total_trainers_envs[env_id]:
                    if trainer in state_samples[env_id]:
                        data_len = len(state_samples[env_id][trainer])
                        value = self.cri_target(state_samples[env_id][trainer])
                        logprob = self.act.get_old_logprob(action_samples[env_id][trainer],
                                                           a_noise_samples[env_id][trainer])
                        last_state = torch.as_tensor((self.last_states[env_id][trainer],),
                                                     dtype=torch.float32, device=self.device)
                        rest_r_sum = self.cri(last_state).detach()
                        r_sum, advantage = self.get_reward_sum(self, data_len, reward_samples[env_id][trainer],
                                                               mask_samples[env_id][trainer], value, rest_r_sum)
                        states.append(state_samples[env_id][trainer])
                        actions.append(action_samples[env_id][trainer])
                        r_sums.append(r_sum)
                        logprobs.append(logprob)
                        advantages.append(advantage)
            states = torch.cat(states)
            actions = torch.cat(actions)
            r_sums = torch.cat(r_sums)
            logprobs = torch.cat(logprobs)
            advantages = torch.cat(advantages)
        return states, actions, r_sums, logprobs, advantages

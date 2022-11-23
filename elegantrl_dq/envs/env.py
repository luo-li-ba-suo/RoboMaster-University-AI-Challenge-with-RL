import robomaster2D
import gym
import numpy as np
from copy import deepcopy
import torch.multiprocessing as mp

gym.logger.set_level(40)


class PreprocessEnv(gym.Wrapper):  # environment wrapper
    def __init__(self, env, if_print=True):
        self.env = gym.make(env) if isinstance(env, str) else env
        super(PreprocessEnv, self).__init__(self.env)
        self.if_multi_processing = False
        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.if_multi_discrete, self.target_return, self.observation_matrix_shape,
         self.total_agents) = get_gym_env_info(self.env, if_print)
        self.args = self.env.args
        self.reset = self.reset_type
        self.step = self.step_type

    def reset_type(self, evaluation=False):
        states, info = self.env.reset(evaluation)
        self.reward_dict = self.env.rewards
        return [np.array(state, dtype=object) for state in states], info

    def step_type(self, actions) -> (list, float, bool, dict):
        states, rewards, done, info = self.env.step(actions)
        self.reward_dict = self.env.rewards
        return [np.array(state, dtype=object) for state in states], \
               [np.array(reward).astype(np.float32) for reward in rewards], done, info

    def display_characters(self, characters):
        self.env.display_characters(characters)


class VecEnvironments:
    def __init__(self, env_name, env_num, pseudo_step=0, seed_offset=0):
        self.env_num = env_num
        self.seed_offset = seed_offset
        print(f"\n{env_num} envs launched \n")
        self.if_multi_processing = True
        self.envs = [PreprocessEnv(env_name, if_print=True if env_id == 0 else False) for env_id in range(env_num)]
        self.state_dim = self.envs[0].state_dim
        self.observation_matrix_shape = self.envs[0].observation_matrix_shape
        self.action_dim = self.envs[0].action_dim
        self.if_discrete = self.envs[0].if_discrete
        self.if_multi_discrete = self.envs[0].if_multi_discrete
        self.target_return = self.envs[0].target_return
        self.env_name = self.envs[0].env_name
        self.args = self.envs[0].env.args
        self.max_step = self.envs[0].max_step
        self.reset_count = [0 for _ in range(env_num)]
        self.pseudo_step = pseudo_step  # 代表在伪终止状态后还需要进行的伪步数

        if env_num > 1:
            self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(env_num)])
            for index in range(env_num):
                process = mp.Process(target=self.run, args=(index,))
                process.start()
                self.env_conns[index].close()

        '''frame stack'''
        self.total_agents = self.envs[0].total_agents
        self.frame_stack_num = 1
        self.history_action_stack_num = 1
        self.if_use_cnn = False
        self.state_origin_dim = self.state_dim
        self.obs_matrix_origin_shape = deepcopy(self.observation_matrix_shape)
        self.init_state_stack = None
        self.history_action_stack_offset = None
        self.states_stack = None

    def init(self, frame_stack_num, history_action_stack_num, if_use_cnn):
        '''帧堆叠'''
        self.frame_stack_num = frame_stack_num
        self.history_action_stack_num = history_action_stack_num
        self.if_use_cnn = if_use_cnn
        self.history_action_stack_offset = self.state_origin_dim * frame_stack_num
        self.state_dim = self.history_action_stack_offset + sum(self.action_dim) * history_action_stack_num
        self.observation_matrix_shape[0] *= frame_stack_num
        if frame_stack_num > 1 or history_action_stack_num > 0:
            if if_use_cnn:
                self.init_state_stack = [{agent: [np.zeros(self.state_dim), np.zeros(self.observation_matrix_shape)]
                                          for agent in self.total_agents} for _ in range(self.env_num)]
            else:
                self.init_state_stack = [{agent: [np.zeros(self.state_dim)]
                                          for agent in self.total_agents} for _ in range(self.env_num)]
            self.states_stack = deepcopy(self.init_state_stack)

    def run(self, index):
        np.random.seed(index + self.seed_offset)
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(self.envs[index].step(action))
            elif request == "reset":
                if action is not None and action['evaluation']:
                    self.env_conns[index].send(self.envs[index].reset(evaluation=True))
                else:
                    self.env_conns[index].send(self.envs[index].reset())
            elif request == "render":
                self.envs[index].render()
            elif request == "display_characters":
                self.envs[index].env.display_characters(action)
            elif request == "stop":
                break
            else:
                raise NotImplementedError

    def render(self, index=0):
        if self.env_num > 1:
            self.agent_conns[index].send(("render", None))
        else:
            self.envs[index].render()

    def display_characters(self, characters, index=0):
        if self.env_num > 1:
            self.agent_conns[index].send(("display_characters", characters))
        else:
            self.envs[index].env.display_characters(characters)

    def reset(self, evaluation=False):
        if self.env_num > 1:
            [agent_conn.send(("reset", {'evaluation': evaluation})) for agent_conn in self.agent_conns]
            curr_states, infos = zip(*[agent_conn.recv() for agent_conn in self.agent_conns])
        else:
            curr_states, infos = self.envs[0].reset(evaluation=evaluation)
            curr_states, infos = [curr_states], [infos]
        '''帧堆叠'''
        if self.frame_stack_num > 1 or self.history_action_stack_num > 0:
            self.states_stack = deepcopy(self.init_state_stack)
            for env_id in range(self.env_num):
                for agent in self.total_agents:
                    for n in range(self.frame_stack_num):
                        if curr_states[env_id][agent] == None:
                            self.states_stack[env_id][agent] = None
                        else:
                            self.states_stack[env_id][agent][0][n * self.state_origin_dim:(n + 1) * self.state_origin_dim] = curr_states[env_id][agent][0]
                            if self.if_use_cnn:
                                self.states_stack[env_id][agent][1][n * self.obs_matrix_origin_shape[0]:(n + 1) * self.obs_matrix_origin_shape[0]] = curr_states[env_id][agent][1]
            curr_states = self.states_stack
        return np.array(curr_states, dtype=object), infos

    def step(self, actions, pseudo_step_flag=True, evaluation=False):
        if self.env_num > 1:
            [agent_conn.send(("step", action)) for agent_conn, action in zip(self.agent_conns, actions)]
            states, rewards, dones, infos = zip(*[agent_conn.recv() for agent_conn in self.agent_conns])
            states = list(states)
            for env_id in range(self.env_num):
                if dones[env_id]:
                    if not infos[env_id]['pseudo_done']:
                        infos[env_id]['pseudo_step_'] = -1
                        self.agent_conns[env_id].send(("reset", {'evaluation': evaluation}))
                        states[env_id], new_info = self.agent_conns[env_id].recv()
                        infos[env_id].update(new_info)
                    else:
                        if self.reset_count[env_id] != self.pseudo_step or not pseudo_step_flag:
                            # 当没有使用动作预测时触发伪终止以及使用动作预测时没有进入伪步时记录
                            # 伪终止状态
                            infos[env_id]['pseudo_terminal_state_'] = np.array(states[env_id], dtype=object)

                        if self.reset_count[env_id] == self.pseudo_step or not pseudo_step_flag:
                            infos[env_id]['pseudo_step_'] = 0
                            self.agent_conns[env_id].send(("reset", {'evaluation': evaluation}))
                            states[env_id], new_info = self.agent_conns[env_id].recv()
                            infos[env_id].update(new_info)
                            self.reset_count[env_id] = 0
                        else:
                            infos[env_id]['pseudo_step_'] = 1  # 代表接下来需要进行伪步
                            self.reset_count[env_id] += 1
        else:
            states, rewards, dones, infos = self.envs[0].step(actions[0])
            states, rewards, dones, infos = [states], [rewards], [dones], [infos]
            if dones[0]:
                if not infos[0]['pseudo_done']:
                    assert self.reset_count[0] == 0, "some bug happens"
                    infos[0]['pseudo_step_'] = -1
                    states[0], new_info = self.envs[0].reset(evaluation=evaluation)
                    infos[0].update(new_info)
                elif infos[0]['pseudo_done']:
                    if self.reset_count[0] != self.pseudo_step or not pseudo_step_flag:
                        # 当没有使用动作预测时触发伪终止以及使用动作预测时没有进入伪步时记录
                        # 伪终止状态
                        infos[0]['pseudo_terminal_state_'] = np.array(states[0], dtype=object)
                    if self.reset_count[0] == self.pseudo_step or not pseudo_step_flag:
                        infos[0]['pseudo_step_'] = 0
                        states[0], new_info = self.envs[0].reset(evaluation=evaluation)
                        infos[0].update(new_info)
                        self.reset_count[0] = 0
                    else:
                        infos[0]['pseudo_step_'] = 1  # 代表接下来需要进行伪步
                        self.reset_count[0] += 1
        '''帧堆叠'''
        if self.frame_stack_num > 1 or self.history_action_stack_num > 0:
            for env_id in range(self.env_num):
                if ('pseudo_step_' in infos[env_id] and infos[env_id]['pseudo_step_'] <= 0) or 'pseudo_step_' not in infos[env_id]:  # 排除伪步;伪步中如果done=True，states中会有None，导致后续处理bug
                    for agent in self.total_agents:
                        if states[env_id][agent] == None:
                            # 此处不能为is None，因为这样判断不出np.array(None)==None
                            # 表示该robot处于死亡状态，无需处理
                            self.states_stack[env_id][agent] = None
                        elif dones[env_id]:
                            if self.states_stack[env_id][agent] is None:
                                self.states_stack[env_id][agent] = [np.zeros(self.state_dim), None]
                                if self.if_use_cnn:
                                    self.states_stack[env_id][agent][1] = np.zeros(self.observation_matrix_shape)
                            # 表示经过了reset，则填充重复的初始状态和空的动作
                            for n in range(self.frame_stack_num):
                                self.states_stack[env_id][agent][0][n * self.state_origin_dim:(n + 1) * self.state_origin_dim] = states[env_id][agent][0]
                                if self.if_use_cnn:
                                    self.states_stack[env_id][agent][1][n * self.obs_matrix_origin_shape[0]:(n + 1) * self.obs_matrix_origin_shape[0]] = states[env_id][agent][1]
                        else:
                            if self.frame_stack_num > 1:
                                # 堆叠超过1则将整体往前移动一层，空出来一层赋予最新的一层
                                self.states_stack[env_id][agent][0][0: (self.frame_stack_num-1) * self.state_origin_dim] = self.states_stack[env_id][agent][0][self.state_origin_dim: self.frame_stack_num * self.state_origin_dim]
                            self.states_stack[env_id][agent][0][(self.frame_stack_num-1) * self.state_origin_dim:self.frame_stack_num * self.state_origin_dim] = states[env_id][agent][0]
                            if self.history_action_stack_num > 1:
                                self.states_stack[env_id][agent][0][self.history_action_stack_offset:self.history_action_stack_offset + (self.history_action_stack_num-1)*sum(self.action_dim)] = self.states_stack[env_id][agent][0][self.history_action_stack_offset+sum(self.action_dim):self.history_action_stack_offset + self.history_action_stack_num*sum(self.action_dim)]
                            if self.history_action_stack_num > 0:
                                self.states_stack[env_id][agent][0][self.history_action_stack_offset + (self.history_action_stack_num - 1) * sum(self.action_dim):self.history_action_stack_offset + self.history_action_stack_num * sum(self.action_dim)] = self.get_one_hot(actions[env_id][agent])
                            if self.if_use_cnn:
                                if self.frame_stack_num > 1:
                                    self.states_stack[env_id][agent][1][0:(self.frame_stack_num-1) * self.obs_matrix_origin_shape[0]] = self.states_stack[env_id][agent][1][self.obs_matrix_origin_shape[0]:self.frame_stack_num * self.obs_matrix_origin_shape[0]]
                                self.states_stack[env_id][agent][1][(self.frame_stack_num-1) * self.obs_matrix_origin_shape[0]:self.frame_stack_num * self.obs_matrix_origin_shape[0]] = states[env_id][agent][1]
                else:
                    for agent in self.total_agents:
                        if states[env_id][agent] != None and infos[env_id]['pseudo_done']:
                            # 针对伪终止状态做帧堆叠
                            state_stack = self.states_stack[env_id][agent]
                            if self.frame_stack_num > 1:
                                state_stack[0][0: (self.frame_stack_num - 1) * self.state_origin_dim] = state_stack[0][self.state_origin_dim: self.frame_stack_num * self.state_origin_dim]
                                if self.if_use_cnn:
                                    state_stack[1][0:(self.frame_stack_num - 1) * self.obs_matrix_origin_shape[0]] = state_stack[1][self.obs_matrix_origin_shape[0]:self.frame_stack_num * self.obs_matrix_origin_shape[0]]
                            state_stack[0][(self.frame_stack_num - 1) * self.state_origin_dim:self.frame_stack_num * self.state_origin_dim] = infos[env_id]['pseudo_terminal_state_'][agent][0]
                            state_stack[1][(self.frame_stack_num - 1) * self.obs_matrix_origin_shape[0]:self.frame_stack_num * self.obs_matrix_origin_shape[0]] = infos[env_id]['pseudo_terminal_state_'][agent][1]
                            infos[env_id]['pseudo_terminal_state_'] = state_stack
            states = self.states_stack

        return np.array(states, dtype=object), np.array(rewards, dtype=object), dones, infos

    def get_one_hot(self, action):
        offset = 0
        one_hot = np.zeros(sum(self.action_dim))
        for i, dim in enumerate(self.action_dim):
            one_hot[offset + action[i]] = 1
            offset += dim
        return one_hot

    def stop(self):
        if self.env_num > 1:
            [agent_conn.send(("stop", None)) for agent_conn in self.agent_conns]
        del self.envs


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
    total_agents = env.total_agents_ids
    observation_matrix_shape = env.observation_matrix_shape
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
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, if_multi_discrete, target_return, observation_matrix_shape, total_agents


def deepcopy_or_rebuild_env(env):
    try:
        env_eval = deepcopy(env)
    except Exception as error:
        print('| deepcopy_or_rebuild_env, error:', error)
        env_eval = PreprocessEnv(env.env_name, if_print=False)
    return env_eval

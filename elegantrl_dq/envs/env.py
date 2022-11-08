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
         self.if_discrete, self.if_multi_discrete, self.target_return, self.observation_matrix_shape) = get_gym_env_info(self.env, if_print)
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
    def __init__(self, env_name, env_num, pseudo_step=0):
        self.env_num = env_num
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

    def run(self, index):
        np.random.seed(index)
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(self.envs[index].step(action))
            elif request == "reset":
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

    def reset(self):
        if self.env_num > 1:
            [agent_conn.send(("reset", None)) for agent_conn in self.agent_conns]
            curr_states, infos = zip(*[agent_conn.recv() for agent_conn in self.agent_conns])
        else:
            curr_states, infos = self.envs[0].reset()
            curr_states, infos = [curr_states], [infos]
        return np.array(curr_states, dtype=object), infos

    def step(self, actions, pseudo_step_flag=True):
        if self.env_num > 1:
            [agent_conn.send(("step", action)) for agent_conn, action in zip(self.agent_conns, actions)]
            states, rewards, dones, infos = zip(*[agent_conn.recv() for agent_conn in self.agent_conns])
            states = list(states)
            for env_id in range(self.env_num):
                if dones[env_id]:
                    if not infos[env_id]['pseudo_done']:
                        infos[env_id]['pseudo_step_'] = -1
                        self.agent_conns[env_id].send(("reset", None))
                        states[env_id], new_info = self.agent_conns[env_id].recv()
                        infos[env_id].update(new_info)
                    else:
                        if self.reset_count[env_id] == self.pseudo_step or not pseudo_step_flag:
                            infos[env_id]['pseudo_step_'] = 0
                            self.agent_conns[env_id].send(("reset", None))
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
                    states[0], new_info = self.envs[0].reset()
                    infos[0].update(new_info)
                elif infos[0]['pseudo_done']:
                    if self.reset_count[0] == self.pseudo_step or not pseudo_step_flag:
                        infos[0]['pseudo_step_'] = 0
                        states[0], new_info = self.envs[0].reset()
                        infos[0].update(new_info)
                        self.reset_count[0] = 0
                    else:
                        infos[0]['pseudo_step_'] = 1  # 代表接下来需要进行伪步
                        self.reset_count[0] += 1
        return np.array(states, dtype=object), np.array(rewards, dtype=object), dones, infos

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
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, if_multi_discrete, target_return, observation_matrix_shape


def deepcopy_or_rebuild_env(env):
    try:
        env_eval = deepcopy(env)
    except Exception as error:
        print('| deepcopy_or_rebuild_env, error:', error)
        env_eval = PreprocessEnv(env.env_name, if_print=False)
    return env_eval

"""
Env_for_RL of ICRA RMUA Environment


Author: DQ, HITSZ
Date: June 8th, 2021

这里开始与强化学习接轨
关于动作解码的部分在rl_trainer.py中
"""
import sys
import gym
import numpy as np
import copy
from robomaster2D.envs import kernel_game
from robomaster2D.envs.options import Parameters

sys.path.append('./robomaster2D/simulator/envs/')
sys.path.append('./robomaster2D/envs/')


class RMUA_Multi_agent_Env(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 200}
    env_name = 'Robomaster'
    target_return = 50

    def __init__(self, args=Parameters()):
        self.do_render = args.render
        self.args = args
        self.robot_num = args.robot_r_num + args.robot_b_num
        self.simulator = kernel_game.Simulator(args)
        self.max_step = args.episode_step if args.episode_step else \
            args.episode_time * args.frame_num_one_second // args.frame_num_one_step

        self.init_obs_space()
        self.init_action_space(args.action_type)

        # reward
        self.reward_range = (float("inf"), float("-inf"))
        self.reward_text = None
        if self.do_render:
            self.reward_text = {}
        self.trainer_ids = []
        self.nn_enemy_ids = []
        for agent in self.simulator.agents:
            if agent.name == 'rl_trainer':
                self.trainer_ids += agent.robot_ids
            if agent.name == 'nn_enemy':
                self.nn_enemy_ids += agent.robot_ids
        self.rewards = [{} for _ in range(self.robot_num)]
        self.rewards_episode = [{} for _ in range(self.robot_num)]
        self.rewards_record = [[] for _ in range(self.robot_num)]

        # env
        self.delta_dist_matrix = [[0 for _ in range(self.simulator.state.robot_num)] for _ in
                                  range(self.simulator.state.robot_num)]
        self.last_time_alive_robots = [True for _ in range(self.robot_num)]
        self.robots_being_killed = []
        # flags
        self.cal_public_obs_already = False

    def init_obs_space(self):
        self.obs_low = []
        self.obs_high = []
        self.obs_set = [{} for _ in range(self.simulator.parameters.robot_num)]
        # 縮小倍數：
        self.obs_set_scale = [[] for _ in range(self.simulator.parameters.robot_num)]

        robot_index = 0
        for agent in self.simulator.agents:
            for n in agent.robot_ids:
                for state_ in agent.state:
                    self.obs_set[robot_index]['robots[' + str(n) + '].' + state_] = agent.state[state_]
                    self.obs_low.append(agent.state[state_][0])
                    self.obs_high.append(agent.state[state_][1])
                    self.obs_set_scale[robot_index].append(agent.state[state_][1])
                robot_index += 1
        # state besides agent(最后有下划线表示不直接应用）
        if self.simulator.state.buff_mode:
            self.public_obs_set = {'time': [0, 180],
                                   'buff1_': [0, 5],
                                   'buff2_': [0, 5],
                                   'buff3_': [0, 5],
                                   'buff4_': [0, 5],
                                   'buff5_': [0, 5],
                                   'buff6_': [0, 5]
                                   }
        else:
            self.public_obs_set = {'time': [0, 180]}
        for n in range(1, self.simulator.state.robot_num):
            self.public_obs_set.update({'dist_' + str(n) + '_': [0, 924],
                                        'x_dist_' + str(n) + '_': [0, 808],
                                        'y_dist_' + str(n) + '_': [0, 448],
                                        'relative_angle_' + str(n) + '_': [-180, 180]
                                        })

        for state_ in self.public_obs_set:
            self.obs_low.append(self.public_obs_set[state_][0])
            self.obs_high.append(self.public_obs_set[state_][1])

        # self.obs_buff = [(buff[0] if buff[1] else 0) for buff in self.simulator.state.buff]
        self.observation_space = gym.spaces.Box(np.array(self.obs_low), np.array(self.obs_high))

    def init_action_space(self, action_type):
        # 合计各个agent的动作空间总和
        if action_type == 'Discrete':
            actions = 1
        else:
            actions = []
        agent = None
        for agent_ in self.simulator.agents:
            if agent_.name == 'rl_trainer':
                agent = agent_
        if agent is not None:
            if action_type == 'MultiDiscrete':
                for action in agent.actions:
                    actions.append(agent.actions[action])
            elif action_type == 'Discrete':
                for action in agent.actions:
                    actions *= agent.actions[action]
            elif action_type == 'Hybrid':
                actions = [[], [[], []]]
                for action in agent.actions['Discrete']:
                    for robot in range(agent.num_robots):
                        actions[0].append(agent.actions['Discrete'][action])
                for action in agent.actions['Continuous'].keys():
                    actions[1][0].append(agent.actions['Continuous'][action][0])
                    actions[1][1].append(agent.actions['Continuous'][action][1])
        if action_type == 'MultiDiscrete':
            # 动作解码在rl_trainer.py中
            self.action_space = gym.spaces.MultiDiscrete(actions)
        elif action_type == 'Discrete':
            # 动作解码在rl_trainer.py中
            self.action_space = gym.spaces.Discrete(np.prod(actions))
        elif action_type == 'Hybrid':
            self.action_space = [gym.spaces.Box(np.array(actions[1][0]), np.array(actions[1][1])),
                                 gym.spaces.MultiDiscrete(actions[0])]

    def reset(self):
        self.trainer_ids = []
        for agent in self.simulator.agents:
            if agent.name == 'rl_trainer':
                self.trainer_ids += agent.robot_ids
        if self.do_render and self.reward_text is None:
            self.reward_text = {}
        self.rewards = [{} for _ in range(self.robot_num)]
        self.rewards_episode = [{} for _ in range(self.robot_num)]
        self.last_dist_matrix = None
        self.simulator.reset()
        self.last_time_alive_robots = [True for _ in range(self.robot_num)]
        for robot in self.simulator.state.robots:
            for key in robot.robot_info_text:
                if '总分' in key:
                    robot.robot_info_text[key] = 0
        for n in self.trainer_ids:
            robot = self.simulator.state.robots[n]
            robot.robot_info_plot['reward'] = self.rewards_record[n]
        return self.get_observations()

    def decode_actions(self, actions):
        if self.args.action_type == 'Discrete':
            i = 0
            actions_ = []
            for agent in self.simulator.agents:
                if agent.actions:
                    for robot in range(agent.num_robots):
                        actions_.append([])
                        action_before_decode = actions[i]
                        i += 1
                        for action_dim in agent.actions:
                            actions_[-1].append(action_before_decode % agent.actions[action_dim])
                            action_before_decode = action_before_decode // agent.actions[action_dim]
            return actions_
        else:
            actions_blank = [[None for _ in range(self.args.robot_r_num)], [None for _ in range(self.args.robot_b_num)]]
            j = 0
            for i in range(self.args.robot_r_num):
                if j < len(actions):
                    actions_blank[0][i] = np.array(actions[j])
                    j += 1
            for i in range(self.args.robot_b_num):
                if j < len(actions):
                    actions_blank[1][i] = np.array(actions[j])
                    j += 1
            return actions_blank

    def step(self, actions):
        done, info = self.simulator.step(self.decode_actions(actions))  # 只给其中一个传动作

        self.robots_being_killed = []
        for i, robot in enumerate(self.simulator.state.robots):
            if robot.hp <= 0 < self.last_time_alive_robots[i]:
                self.last_time_alive_robots[i] = False
                self.robots_being_killed.append(i)

        if done:
            info_dicts = {'win_rate': self.simulator.state.r_win_record.get_win_rate(),
                          'draw_rate': self.simulator.state.r_win_record.get_draw_rate(),
                          'robots_being_killed_': self.robots_being_killed}  # it's not recorded if its key ends with _
        else:
            info_dicts = {'robots_being_killed_': self.robots_being_killed}
        info_dicts.update(info)

        r = self.compute_reward()
        # 记录每个机器人每回合的奖励：
        if done and self.do_render:
            for n in self.trainer_ids:
                self.rewards_record[n].append(sum(self.rewards_episode[n].values()))
                if len(self.rewards_record[n]) > 500:  # 如果超过500条记录就均匀减半
                    self.rewards_record[n] = self.rewards_record[n][::2]
        # trainer死亡的时刻还应该计算一次奖励，所以要先计算奖励后删除trainer:
        for i, trainer in enumerate(self.trainer_ids):
            if self.if_trainer_dead(trainer):
                del self.trainer_ids[i]

        return self.get_observations(), r, done, info_dicts

    def compute_reward(self):
        rewards = []
        for n in self.trainer_ids:
            robot = self.simulator.state.robots[n]
            # '''血量减少'''
            # reward -= 0.05 * robot.hp_loss.one_step
            # '''拿到补给'''
            # reward += 0.05 * robot.buff_hp.one_step
            # reward += 0.05 * robot.buff_bullet.one_step
            '''消耗子弹'''
            # self.rewards[n]['bullet_out'] = -0.005 * robot.bullet_out_record.one_step
            '''hit_enemy'''
            self.rewards[n]['hit'] = 0
            self.rewards[n]['hit'] += 2 * robot.enemy_hit_record.left.one_step
            self.rewards[n]['hit'] += 2 * robot.enemy_hit_record.right.one_step
            self.rewards[n]['hit'] += 5 * robot.enemy_hit_record.behind.one_step
            self.rewards[n]['hit'] += 1 * robot.enemy_hit_record.front.one_step
            # '''被敌军击中'''
            self.rewards[n]['hit_by_enemy'] = 0
            self.rewards[n]['hit_by_enemy'] -= 2 * robot.armor_hit_enemy_record.left.one_step
            self.rewards[n]['hit_by_enemy'] -= 2 * robot.armor_hit_enemy_record.right.one_step
            self.rewards[n]['hit_by_enemy'] -= 5 * robot.armor_hit_enemy_record.behind.one_step
            self.rewards[n]['hit_by_enemy'] -= 1 * robot.armor_hit_enemy_record.front.one_step
            # '''击中友军'''
            # reward -= 0.005 * robot.teammate_hit_record.left.one_step
            # reward -= 0.005 * robot.teammate_hit_record.right.one_step
            # reward -= 0.01 * robot.teammate_hit_record.behind.one_step
            # reward -= 0.002 * robot.teammate_hit_record.front.one_step
            # '''轮子撞墙、撞机器人'''
            self.rewards[n]['wheel_hit'] = 0
            self.rewards[n]['wheel_hit'] -= 1 * robot.wheel_hit_obstacle_record.one_step
            self.rewards[n]['wheel_hit'] -= 1 * robot.wheel_hit_wall_record.one_step
            self.rewards[n]['wheel_hit'] -= 1 * robot.wheel_hit_robot_record.one_step
            # '''装甲板撞墙'''
            # self.rewards[n]['hit_by_wall'] = 0
            # self.rewards[n]['hit_by_wall'] -= 0.05 * robot.armor_hit_wall_record.left.one_step
            # self.rewards[n]['hit_by_wall'] -= 0.05 * robot.armor_hit_wall_record.right.one_step
            # self.rewards[n]['hit_by_wall'] -= 0.1 * robot.armor_hit_wall_record.behind.one_step
            # self.rewards[n]['hit_by_wall'] -= 0.02 * robot.armor_hit_wall_record.front.one_step
            self.rewards[n]['hit_by_obstacle'] = 0
            self.rewards[n]['hit_by_obstacle'] -= 0.5 * robot.armor_hit_obstacle_record.left.one_step
            self.rewards[n]['hit_by_obstacle'] -= 0.5 * robot.armor_hit_obstacle_record.right.one_step
            self.rewards[n]['hit_by_obstacle'] -= 1 * robot.armor_hit_obstacle_record.behind.one_step
            self.rewards[n]['hit_by_obstacle'] -= 0.2 * robot.armor_hit_obstacle_record.front.one_step
            # '''装甲板撞机器人'''
            self.rewards[n]['hit_by_robot'] = 0
            self.rewards[n]['hit_by_robot'] -= 0.5 * robot.armor_hit_robot_record.left.one_step
            self.rewards[n]['hit_by_robot'] -= 0.5 * robot.armor_hit_robot_record.right.one_step
            self.rewards[n]['hit_by_robot'] -= 1 * robot.armor_hit_robot_record.behind.one_step
            self.rewards[n]['hit_by_robot'] -= 0.2 * robot.armor_hit_robot_record.front.one_step
            '''过热惩罚'''
            # reward -= 0.005 * robot.hp_loss_from_heat.one_step
            '''no_move惩罚'''
            # self.rewards[n]['no_move'] = -1 if robot.vx == 0 and robot.vy == 0 else 0
            '''击杀对方奖励'''
            enemy_all_defeated = True
            enemy_defeated = False
            for j, enemy_id in enumerate(robot.enemy):
                if enemy_id in self.robots_being_killed:
                    enemy_defeated = True
                if self.simulator.state.robots[enemy_id].hp > 0:
                    enemy_all_defeated = False
            if enemy_all_defeated:
                self.rewards[n]['K.O.'] = 200
            elif enemy_defeated:
                self.rewards[n]['K.O.'] = 100
            else:
                self.rewards[n]['K.O.'] = 0
                # '''引导：进攻模式'''
                # '''离敌人越近负奖励越小'''
                # dist = self.simulator.state.dist_matrix[n][enemy_id]
                # delta_dist = self.delta_dist_matrix[n][enemy_id]
                # self.rewards[n]['chase'] = -delta_dist * 0.1 if dist > 250 else delta_dist * 0.1

            reward = 0
            for key in self.rewards[n]:
                reward += self.rewards[n][key]
                robot.robot_info_text[key + '得分'] = self.rewards[n][key]
                if key + '总分' in robot.robot_info_text:
                    robot.robot_info_text[key + '总分'] += self.rewards[n][key]
                else:
                    robot.robot_info_text[key + '总分'] = self.rewards[n][key]
                if key in self.rewards_episode[n]:
                    self.rewards_episode[n][key] += self.rewards[n][key]
                else:
                    self.rewards_episode[n][key] = self.rewards[n][key]
            robot.robot_info_text['得分'] = reward
            if '总分' in robot.robot_info_text:
                robot.robot_info_text['总分'] += reward
            else:
                robot.robot_info_text['总分'] = reward
            rewards.append(reward)
        return rewards

    def calculate_public_observation(self):
        game_state = self.simulator.state
        if self.last_dist_matrix:
            for n, robot in enumerate(game_state.robots):
                for i, other_robot in enumerate(game_state.robots):
                    if n != i:
                        self.delta_dist_matrix[n][i] = self.simulator.state.dist_matrix[n][i] - \
                                                       self.last_dist_matrix[n][i]  # 代表与敌人距离的增加
        self.last_dist_matrix = copy.deepcopy(self.simulator.state.dist_matrix)

        self.public_observation = []
        for i, state in enumerate(self.public_obs_set):
            if state[-1] != '_':
                self.public_observation.append(eval('game_state.' + state) / self.public_obs_set[state][1])

        # 针对buff单独读取观测值

        if self.simulator.state.buff_mode:
            for buff_area in game_state.buff:
                if buff_area[1]:
                    self.public_observation.append((buff_area[0] + 1) / 6)  # 加一是为了区分未激活编号为0的buff和已激活buff
                else:
                    self.public_observation.append(0)
        self.cal_public_obs_already = True

    def get_individual_observation(self, robot_index):
        robot = self.simulator.state.robots[robot_index]
        # 在使用該函數前須先運行get_public_observation函數
        assert self.cal_public_obs_already, 'Please run get_public_observation function first'

        observation = []

        # 友方信息放前面，敵方信息放後面
        # 自己的信息
        for i, state in enumerate(self.obs_set[robot_index]):
            observation.append(eval('self.simulator.state.' + state) / self.obs_set_scale[robot_index][i])
            robot.robot_info_text[state] = observation[i]
        # 友方信息
        if robot.friend is not None:
            for i, state in enumerate(self.obs_set[robot.friend]):
                observation.append(eval('self.simulator.state.' + state) / self.obs_set_scale[robot.friend][i])
        # 敵方信息
        for enemy_id in robot.enemy:
            for i, state in enumerate(self.obs_set[enemy_id]):
                observation.append(eval('self.simulator.state.' + state) / self.obs_set_scale[enemy_id][i])
        # 额外部分
        observation += self.public_observation
        # 相對距离
        # 友方
        if robot.friend is not None:
            observation.append(self.simulator.state.dist_matrix[robot_index][robot.friend] / 853568)
            observation.append(self.simulator.state.x_dist_matrix[robot_index][robot.friend] / 808)
            observation.append(self.simulator.state.y_dist_matrix[robot_index][robot.friend] / 448)
            # 相对角度
            observation.append((self.simulator.state.relative_angle[robot_index, robot.friend]) / 180)
        # 敵方
        for i in robot.enemy:
            observation.append(self.simulator.state.dist_matrix[robot_index][i] / 853568)
            observation.append(self.simulator.state.x_dist_matrix[robot_index][i] / 808)
            observation.append(self.simulator.state.y_dist_matrix[robot_index][i] / 448)
            # 相对角度
            observation.append((self.simulator.state.relative_angle[robot_index, i]) / 180)

        return [np.array(observation).astype(np.float32), self.simulator.state.robots[robot_index].local_map.astype(np.float32)]

    def get_observations(self):
        self.calculate_public_observation()
        observations = []
        for i in range(self.args.robot_r_num + self.args.robot_b_num):
            if i in self.trainer_ids or i in self.nn_enemy_ids:
                observations.append(self.get_individual_observation(i))
            else:
                observations.append(None)  # for random agent
        self.cal_public_obs_already = False
        return observations

    def render(self, mode='human'):
        if not self.simulator.render_inited and self.simulator.parameters.render:
            self.simulator.init_render(self.args)

    def display_characters(self, characters='', position='title'):
        if self.simulator.render_inited:
            self.simulator.module_UI.text_training_state = characters

    def if_trainer_dead(self, idx):
        if_dead = self.simulator.state.robots[idx].hp == 0
        assert self.simulator.state.robots[idx].hp >= 0, f"Warning: robot {idx} hp < 0!"
        return if_dead


if __name__ == '__main__':
    args = Parameters()
    args.red_agents_path = 'src.agents.human_agent'
    args.blue_agents_path = 'src.agents.handcrafted_enemy'
    args.render_per_frame = 20
    args.episode_step = 0
    args.render = True
    args.training_mode = False
    args.time_delay_frame = 0.1
    env = RMUA_Multi_agent_Env(args)
    env.simulator.state.pause = True
    env.reset()
    env.render()
    for e in range(args.episodes):
        _, _, done, _ = env.step([])
        if done:
            env.reset()

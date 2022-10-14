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
        self.individual_observations = None
        self.public_observation = None

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
        self.tester_ids = []
        for agent in self.simulator.agents:
            if agent.name == 'rl_trainer':
                self.trainer_ids += agent.robot_ids
            if agent.name == 'nn_enemy':
                self.tester_ids += agent.robot_ids
        self.rewards = [{} for _ in range(self.robot_num)]
        self.rewards_episode = [{} for _ in range(self.robot_num)]
        self.rewards_record = [[] for _ in range(self.robot_num)]

        self.observation_matrix_shape = [1, args.local_map_size, args.local_map_size]
        # flags
        self.cal_public_obs_already = False

    def init_obs_space(self):
        self.obs_low = []
        self.obs_high = []
        for agent in self.simulator.agents:
            ids = agent.get_robot_ids()
            for n in ids:
                obs_low_, obs_high_ = agent.get_individual_obs_space(n)
                self.obs_low += obs_low_
                self.obs_high += obs_high_

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
            self.public_obs_set = {}
        for n in range(1, self.simulator.state.robot_num):
            self.public_obs_set.update({'dist_' + str(n) + '_': [0, 924],
                                        'x_dist_' + str(n) + '_': [0, 808],
                                        'y_dist_' + str(n) + '_': [0, 448],
                                        'relative_angle_' + str(n) + '_': [-180, 180]
                                        })

        for state_ in self.public_obs_set:
            self.obs_low.append(self.public_obs_set[state_][0])
            self.obs_high.append(self.public_obs_set[state_][1])

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
        self.simulator.reset()

        self.trainer_ids = []
        self.tester_ids = []
        for agent in self.simulator.agents:
            if agent.name == 'rl_trainer':
                self.trainer_ids += agent.robot_ids
            if agent.name == 'nn_enemy':
                self.tester_ids += agent.robot_ids
        if self.do_render and self.reward_text is None:
            self.reward_text = {}
        self.rewards = [{} for _ in range(self.robot_num)]
        self.rewards_episode = [{} for _ in range(self.robot_num)]
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

        r = self.compute_reward()
        # 记录每个机器人每回合的奖励：
        if done and self.do_render:
            for n in self.trainer_ids:
                self.rewards_record[n].append(sum(self.rewards_episode[n].values()))
                if len(self.rewards_record[n]) > 500:  # 如果超过500条记录就均匀减半
                    self.rewards_record[n] = self.rewards_record[n][::2]
        obs = self.get_observations()  # 在删除当前死亡的机器人之前最后一次返回其观测向量
        # trainer死亡的时刻还应该计算一次奖励，所以要先计算奖励后删除trainer:
        for i in info['robots_being_killed_']:
            if i in self.trainer_ids:
                self.trainer_ids.remove(i)
            elif i in self.tester_ids:
                self.tester_ids.remove(i)
        return obs, r, done, info

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
            self.rewards[n]['no_move'] = -10 if robot.vx == 0 and robot.vy == 0 else 0
            '''死亡惩罚'''
            self.rewards[n]['death'] = 0
            if n in self.simulator.state.robots_killed_this_step:
                self.rewards[n]['death'] -= 50
            if robot.friend is not None:
                if robot.friend in self.simulator.state.robots_killed_this_step:
                    self.rewards[n]['death'] -= 50
            '''击杀对方奖励'''
            self.rewards[n]['K.O.'] = 0
            enemy_all_defeated = True
            for j, enemy_id in enumerate(robot.enemy):
                if enemy_id in self.simulator.state.robots_killed_this_step:
                    self.rewards[n]['K.O.'] += 100
                if self.simulator.state.robots_survival_status[enemy_id]:
                    enemy_all_defeated = False
            if enemy_all_defeated:
                self.rewards[n]['K.O.'] += 100
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
        self.public_observation = []
        for i, state in enumerate(self.public_obs_set):
            if state[-1] != '_':
                self.public_observation.append(eval('self.simulator.state.' + state) / self.public_obs_set[state][1])

        self.individual_observations = []
        for agent in self.simulator.agents:
            for idx in agent.get_robot_ids():
                self.individual_observations.append(agent.get_individual_obs(idx, self.simulator.state))
        # 针对buff单独读取观测值

        if self.simulator.state.buff_mode:
            for buff_area in self.simulator.state.buff:
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
        observation += self.public_observation

        # 友方信息放前面，敵方信息放後面
        # 自己的信息
        observation += self.individual_observations[robot_index]
        # 友方信息
        if robot.friend is not None:
            observation += self.individual_observations[robot.friend]
        # 敵方信息
        for enemy_id in robot.enemy:
            observation += self.individual_observations[enemy_id]
        # 相對距离
        # 友方
        if robot.friend is not None:
            if self.if_robot_valid(robot.friend):
                observation.append(0.0)
                observation.append(0.0)
                observation.append(0.0)
                observation.append(0.0)
            else:
                observation.append(self.simulator.state.dist_matrix[robot_index][robot.friend] / 924)
                observation.append(self.simulator.state.x_dist_matrix[robot_index][robot.friend] / 808)
                observation.append(self.simulator.state.y_dist_matrix[robot_index][robot.friend] / 448)
                # 相对角度
                observation.append((self.simulator.state.relative_angle[robot_index, robot.friend]) / 180)
        # 敵方
        for i in robot.enemy:
            if self.if_robot_valid(i):
                observation.append(0.0)
                observation.append(0.0)
                observation.append(0.0)
                observation.append(0.0)
            else:
                observation.append(self.simulator.state.dist_matrix[robot_index][i] / 924)
                observation.append(self.simulator.state.x_dist_matrix[robot_index][i] / 808)
                observation.append(self.simulator.state.y_dist_matrix[robot_index][i] / 448)
                # 相对角度
                observation.append((self.simulator.state.relative_angle[robot_index, i]) / 180)

        return [np.array(observation).astype(np.float32),
                self.simulator.state.robots[robot_index].local_map.astype(np.float32)]

    def if_robot_valid(self, idx):  # 仅当机器人存活或刚被击杀时有效
        if self.simulator.state.robots_survival_status[idx] or idx in self.simulator.state.robots_killed_this_step:
            return True
        else:
            return False

    def get_observations(self):
        self.calculate_public_observation()
        observations = []
        for i in range(self.args.robot_r_num + self.args.robot_b_num):
            if i in self.trainer_ids or i in self.tester_ids:
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

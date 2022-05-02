"""
kernel_game of ICRA RMUA Environment


Author: DQ, HITSZ
Date: June 8th, 2021
"""
import numpy as np
from simulator.envs.kernel_map import Map
from simulator.envs.kernel_engine import Engine
from simulator.envs.kernel_objects import Robot
from simulator.envs.kernel_referee import Referee
from simulator.envs.src.agents.human_agent import Orders_set, My_Agent
from itertools import combinations
import importlib
import traceback
import sys
import time


def loadAgents(args):
    file_list = [args.red_agents_path, args.blue_agents_path]
    name_list = [args.red_agents_name, args.blue_agents_name]
    agents = [None, None]
    load_errs = [[], []]
    print(f"path is {file_list}")
    for i, agent_file_path in enumerate(file_list):
        agent_file_path = 'simulator.envs.' + agent_file_path
        agent_temp = None
        print(f"path is {agent_file_path}")

        try:
            my_module = importlib.import_module(agent_file_path)
            agent_temp = my_module.My_Agent(i, args)
        except (NameError, ImportError, IOError):
            print('Error: The team "' + agent_file_path + '" could not be loaded! ', file=sys.stderr)
            traceback.print_exc()
            pass
        except BaseException as e:
            print(e)
            pass

        # if student's agent does not exist, use random agent.
        if agent_temp is not None:
            agents[i] = agent_temp
            if not args.superQuiet:
                print('Agent {} team {} agent {} loaded'.format(i, name_list[i], file_list[i]))
        else:
            agents[i] = My_Agent(i, args)
            load_errs[i].append('[Error] Agent {} team {} agent {} cannot be loaded \n, load human agent instead.' \
                                .format(i, name_list[i], ".".join((file_list[i]).split(".")[-2:])))
    return agents, load_errs


class State(object):  # 总状态
    def __init__(self, options):
        self.do_render = options.render
        self.time = 0  # 比赛剩余时间
        self.frame = 0
        self.step = 0
        self.episode = -1
        self.robots = []  # 车的状态
        self.robot_r_num = options.robot_r_num
        self.robot_b_num = options.robot_b_num
        self.robot_num = options.robot_r_num + options.robot_b_num
        self.bullets = []
        self.buff = None
        self.done = True  # 比赛是否结束
        self.lidar_detect = None  # 激光雷达感應到的8個角度的可活動距離
        self.lidar_num = options.lidar_num
        self.camera_vision = None  # 摄像头能看到的车
        self.relative_angle = None  # 相對角度
        self.dist_matrix = [[0 for _ in range(self.robot_num)] for _ in
                            range(self.robot_num)]  # 距離矩陣
        self.x_dist_matrix = [[0 for _ in range(self.robot_num)] for _ in
                              range(self.robot_num)]  # 距離矩陣
        self.y_dist_matrix = [[0 for _ in range(self.robot_num)] for _ in
                              range(self.robot_num)]  # 距離矩陣
        self.goals = None
        self.buff_mode = options.buff_mode

        self.no_dying = options.no_dying

        # 按键、鼠标事件
        self.KB_events = []
        # 是否暂停
        self.pause = False
        # 动画帧率
        self.render_per_frame = options.render_per_frame
        # 是否暂停运行等待用户指令
        self.wait_for_user_input = False

    def reset(self, time, start_pos, start_angle, start_bullet, start_hp):
        self.episode += 1
        self.time = time  # 比赛剩余时间
        self.frame = 0
        self.step = 0
        self.robots = []  # 车的状态
        self.bullets = []
        self.buff = None
        self.done = False  # 比赛是否结束
        self.camera_vision = np.zeros((self.robot_num, self.robot_num), dtype='int8')
        self.lidar_detect = np.zeros((self.robot_num, self.lidar_num), dtype='float64')
        self.relative_angle = np.zeros((self.robot_num, self.robot_num), dtype='float64')

        for n in range(self.robot_r_num):
            self.robots.append(Robot(self.robot_r_num, self.robot_num, 0, n, x=start_pos[n][0],
                                     y=start_pos[n][1],
                                     angle=start_angle[n],
                                     bullet=start_bullet[n], no_dying=self.no_dying, hp=start_hp[n]))
        for n in range(self.robot_b_num):
            self.robots.append(Robot(self.robot_r_num, self.robot_num, 1, n, x=start_pos[n + 2][0],
                                     y=start_pos[n + 2][1],
                                     angle=start_angle[n + 2],
                                     bullet=start_bullet[n + 2], no_dying=self.no_dying, hp=start_hp[n + 2]))

        if self.buff_mode:
            self.random_buff_info()

    def random_buff_info(self):  # 随机分配buff位置，第二列表示是否被使用
        # reset the buff
        tmp_buff_info = np.zeros((6, 2), dtype=np.int16)
        a = [0, 1, 2]
        b = [0, 1]
        c = [[0, 2], [1, 3], [4, 5]]
        np.random.shuffle(a)
        for ia, tmp_a in enumerate(a):
            np.random.shuffle(b)
            for ib, tmp_b in enumerate(b):
                tmp_buff_info[ia + 3 * ib] = (c[tmp_a][tmp_b], 1)
        self.buff = tmp_buff_info


class Parameters(object):  # 参数集合
    def __init__(self, options):
        self.episode_time = options.episode_time
        self.robot_r_num = options.robot_r_num
        self.robot_b_num = options.robot_b_num
        self.robot_num = options.robot_r_num + options.robot_b_num
        self.impact_effect = options.impact_effect
        self.buff_mode = options.buff_mode
        self.unlimited_bullet = options.unlimited_bullet
        self.do_route_plan = options.do_route_plan
        self.rotate_by_route_plan = options.rotate_by_route_plan
        # self.start_pos = [[758, 398], [758, 50], [50, 50], [50, 398]]
        self.start_pos = options.start_pos
        self.random_start_pos = options.random_start_pos
        self.start_angle = options.start_angle
        self.start_bullet = options.start_bullet
        self.start_hp = options.start_hp
        self.frame_num_one_time = options.frame_num_one_time
        self.frame_num_one_second = options.frame_num_one_second
        self.episode_step = options.episode_step
        self.random_start_far_pos = options.random_start_far_pos
        self.do_plot = options.do_plot
        self.collision_bounce = options.collision_bounce

        self.enable_blocks = options.enable_blocks

    def random_set_start_pos(self, positions):
        self.start_pos = []
        self.start_angle = []
        if self.enable_blocks:
            indexs = np.random.choice(range(0, len(positions)), 4, replace=False)
            if self.random_start_far_pos:
                while np.linalg.norm(np.array(positions[indexs[0]]) - np.array(positions[indexs[2]])) < 500:
                    indexs = np.random.choice(range(0, len(positions)), 4, replace=False)

            for i in indexs:
                self.start_pos.append(positions[i])
        else:
            poses = []
            for i in range(4):
                poses.append([np.random.uniform(30, 778), np.random.uniform(30, 418)])
            while np.linalg.norm(np.array(poses[0]) - np.array(poses[2])) < 500:
                poses = []
                for i in range(4):
                    poses.append([np.random.uniform(30, 778), np.random.uniform(30, 418)])
            self.start_pos = poses
        for i in range(4):
            self.start_angle.append(np.random.random() * 360 - 180)


class Simulator(object):
    def __init__(self, options):
        self.parameters = Parameters(options)  # parameters
        self.map = Map(options)  # the map
        self.state = State(options)  # initial state
        self.module_referee = Referee(self.state, self.map, options)  # the referee
        self.module_engine = Engine(self.state, self.module_referee, options, self.map)  # the controller
        self.orders = Orders_set((options.robot_r_num + options.robot_b_num))
        self.combination_robot_id = list(combinations(range(self.state.robot_num), 2))
        self.agents, _ = loadAgents(options)
        self.render_inited = False
        self.state.reset(self.parameters.episode_time, self.parameters.start_pos,
                         self.parameters.start_angle, self.parameters.start_bullet, self.parameters.start_hp)
        self.render_frame = 0  # 渲染时清零
        if self.state.do_render:
            self.init_render(options)

        # 手动调试内容：
        self.single_input = options.single_input
        # 每一帧的延迟
        self.delay_per_frame = options.time_delay_frame

    def init_render(self, options):
        from kernel_user_interface import User_Interface
        self.module_UI = User_Interface(
            self.state, self.module_engine, self.orders, self.map, options)
        self.render_inited = True

    def reset(self):
        self.step_num = 0
        if self.parameters.random_start_pos:
            self.parameters.random_set_start_pos(self.map.goal_positions)
        self.state.reset(self.parameters.episode_time, self.parameters.start_pos,
                         self.parameters.start_angle, self.parameters.start_bullet, self.parameters.start_hp)
        self.orders.reset()
        # 清空上一次eposode记录
        for robot in self.state.robots:
            robot.reset_episode()
        # 启动摄像头、雷达
        self.run_camera_lidar()
        return self.state

    def step(self, actions):  # 执行多智能体动作
        if self.state.do_render:
            self.step_feedback_UI()
        # 判断游戏是否暂停或等待用户输入指令
        if not self.state.pause and not self.state.wait_for_user_input:
            # 如果启动了就开始调用agent解码指令
            self.orders.combine(self.agents[0].decode_actions(self.state, actions[0]),
                                self.agents[1].decode_actions(self.state, actions[1]))
            # 动作输入为None时表示由键盘控制
            self.step_reset()
            for n in range(self.parameters.frame_num_one_time):
                if self.one_frame():  # 一个周期，5ms
                    return True
            self.state.step += 1
            if self.single_input:
                self.state.wait_for_user_input = True
        if self.state.do_render:
            self.render()
        # episode_step如果不为零，当step数量达到这个值，将提前结束episode
        if self.parameters.episode_step:
            if self.state.step == self.parameters.episode_step:
                return True
        return False

    #
    # def step_orders(self, orders):
    #     self.step_feedback_UI()
    #     # 如果启动了就开始解码指令
    #     if not self.state.pause:
    #         self.orders.combine(*orders)
    #         self.step_reset()
    #     for n in range(self.parameters.frame_num_one_time):
    #         if self.one_frame():  # 一个周期，5ms
    #             return True
    #     return False

    def step_feedback_UI(self):
        # 接受并处理UI的反馈信息
        if True in self.module_UI.feedback_UI['reset']:
            self.reset()
            self.module_UI.feedback_UI['reset'] = []
        if True in self.module_UI.feedback_UI['reset_frame']:
            self.module_UI.feedback_UI['reset_frame'] = []
            self.state.frame = 0
        if self.module_UI.feedback_UI['continue']:
            self.state.wait_for_user_input = False
            self.module_UI.feedback_UI['continue'] = False

    def step_reset(self):
        # 清空当前step记录
        for robot in self.state.robots:
            robot.reset_step()

    '''
    ----one_frame函数----
    该函数运行游戏的一帧，返回是否done和是否pause的布尔值
    '''

    def one_frame(self):
        time.sleep(self.delay_per_frame)
        # 清空单帧信息
        for robot in self.state.robots:
            robot.reset_frame()
        # 计时结束
        if self.state.time == 0:
            # 此处不必将self.state.frame置零，因为self.state.reset函数会做这一项工作
            return True
        # 计时
        if not self.state.frame % self.parameters.frame_num_one_second:  # 1frame = 0.005s ~ 200Hz
            self.state.time -= 1
            # 随机分配buff位置
            if not self.state.time % 60:
                if self.state.buff_mode:
                    self.state.random_buff_info()
        # 如果时间未到，则直接返回
        if self.state.time > 180:
            self.module_UI.update_display()
            return False
        # 启动摄像头、雷达
        self.run_camera_lidar()
        # 启动引擎，计算一帧下机器人和子弹的运动
        act_feedback = self.module_engine.one_frame(self.orders)
        # 50HZ 检测装甲板是否要扣血
        if not self.state.frame % 4:
            self.module_referee.check_armor(act_feedback)
        # 10HZ 冷却枪管
        self.module_referee.cooling_barrel()
        # 无延迟检测buff
        if self.state.buff_mode:
            self.module_referee.buff_check()
        # 更新hp
        for robot in self.state.robots:
            robot.update_hp()
        # 帧数+1
        self.state.frame += 1
        return False

    def render(self):
        # 用户界面：
        if self.state.do_render:
            # 检测交互事件
            self.module_UI.update_events()
            # 刷新画面
            if not self.render_frame % self.state.render_per_frame:
                self.module_UI.update_display()
                self.render_frame = 0
        self.render_frame += 1

    def run_camera_lidar(self):
        for n in range(self.state.robot_num):
            for i in range(self.state.robot_num - 1):
                x, y = np.array(self.state.robots[n - i - 1].center) - np.array(self.state.robots[n].center)
                angle = np.angle(x + y * 1j, deg=True)
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                # 计算机身相对与敌方中心连线的偏离角度
                angle = angle - self.state.robots[n].angle
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                self.state.relative_angle[n, n - i - 1] = angle
                camera_can_see = abs(angle) < self.state.robots[n].camera_angle
                obstacle_block = self.module_referee.line_barriers_check(self.state.robots[n].center,
                                                                         self.state.robots[n - i - 1].center) \
                                 or self.module_referee.line_robots_check(self.state.robots[n].center,
                                                                          self.state.robots[n - i - 1].center)
                dist = (x ** 2 + y ** 2) ** 0.5
                self.state.x_dist_matrix[n][n - i - 1] = -x
                self.state.y_dist_matrix[n][n - i - 1] = -y
                self.state.dist_matrix[n][n - i - 1] = dist
                if obstacle_block:
                    self.state.camera_vision[n, n - i - 1] = 0
                else:
                    if camera_can_see:
                        self.state.camera_vision[n, n - i - 1] = 1
                    else:
                        self.state.camera_vision[n, n - i - 1] = 0
            for angle_idx in range(self.state.lidar_num):
                angle = 2 * np.pi / self.state.lidar_num * angle_idx
                # TODO: 計算雷達距離

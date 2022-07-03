"""
kernel_game of ICRA RMUA Environment


Author: DQ, HITSZ
Date: June 8th, 2021
"""
import numpy as np
from robomaster2D.envs.kernel_map import Map
from robomaster2D.envs.kernel_engine import Engine
from robomaster2D.envs.kernel_objects import Robot
from robomaster2D.envs.kernel_referee import Referee
from robomaster2D.envs.src.agents.human_agent import Orders_set, My_Agent
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
        agent_file_path = 'robomaster2D.envs.' + agent_file_path
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


class Alarm20hz(object):
    def __init__(self, frame_num_one_second):
        # 由于装甲板检测受击的频率为20hz
        # 20hz的闹铃频率
        # 如果没有接收到闹铃响起将会堆积
        self.alarm_interval_frame = frame_num_one_second // 20
        self.delta_frame = 0
        self.last_frame = -1
        self.go_off_flag = False

    def go_off(self, frame):
        assert frame >= self.last_frame, "alarm 20hz not reset"
        if frame == self.last_frame:  # 使得在同一帧中可以重复使用
            return self.go_off_flag
        self.delta_frame += frame - self.last_frame
        self.last_frame = frame
        if self.delta_frame >= self.alarm_interval_frame:
            self.delta_frame -= self.alarm_interval_frame
            self.go_off_flag = True
            return True
        else:
            self.go_off_flag = False
            return False


class WinRateManager(object):
    def __init__(self):
        self.record = []
        self.win_num = 0
        self.fail_num = 0
        self.draw_num = 0

    def win(self):
        self.record.append(1)
        self.win_num += 1
        self.hold_total_num()

    def fail(self):
        self.record.append(-1)
        self.fail_num += 1
        self.hold_total_num()

    def draw(self):
        self.record.append(0)
        self.draw_num += 1
        self.hold_total_num()

    def get_win_rate(self):
        if not self.record:
            return 0
        return self.win_num / len(self.record)

    def get_draw_rate(self):
        if not self.record:
            return 0
        return self.draw_num / len(self.record)

    def hold_total_num(self):
        if len(self.record) > 100:
            if self.record[0] == 1:
                self.win_num -= 1
            elif self.record[0] == -1:
                self.fail_num -= 1
            elif self.record[0] == 0:
                self.draw_num -= 1
            del self.record[0]


class State(object):  # 总状态
    def __init__(self, options):
        self.frame_num_one_second = options.frame_num_one_second
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
        # 20hz的闹钟
        self.alarm20hz = Alarm20hz(self.frame_num_one_second)

        self.r_win_record = WinRateManager()

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
        # 20hz的闹钟
        self.alarm20hz = Alarm20hz(self.frame_num_one_second)

        for n in range(self.robot_r_num):
            self.robots.append(Robot(self.robot_r_num, self.robot_num, 0, n, x=start_pos[n][0],
                                     y=start_pos[n][1],
                                     angle=start_angle[n],
                                     bullet=start_bullet[n], no_dying=self.no_dying, hp=start_hp[n]))
        for n in range(self.robot_b_num):
            self.robots.append(Robot(self.robot_r_num, self.robot_num, 1, n, x=start_pos[n + self.robot_r_num][0],
                                     y=start_pos[n + self.robot_r_num][1],
                                     angle=start_angle[n + self.robot_r_num],
                                     bullet=start_bullet[n + self.robot_r_num], no_dying=self.no_dying,
                                     hp=start_hp[n + self.robot_r_num]))

        if self.buff_mode:
            self.random_buff_info()

    def tick(self):
        # 清空单帧信息
        for robot in self.robots:
            robot.reset_frame()
        # 计时
        if not self.frame % self.frame_num_one_second:  # 1frame = 0.05s ~ 20Hz
            self.time -= 1
            # 随机分配buff位置
            if not self.time % 60:
                if self.buff_mode:
                    self.random_buff_info()

    def finish_tick(self):
        # 更新hp
        for robot in self.robots:
            robot.update_hp()
        # 帧数+1
        self.frame += 1

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

    def if_alarm20hz_goes_off(self):
        return self.alarm20hz.go_off(self.frame)

    def if_end(self):
        if self.time <= 0:
            return True

    def if_not_started_yet(self):
        if self.time > 180:
            return True


class Parameters(object):  # 参数集合
    def __init__(self, options):
        self.render = options.render
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
        self.frame_num_one_step = options.frame_num_one_step
        self.frame_num_one_second = options.frame_num_one_second
        self.episode_step = options.episode_step
        self.random_start_far_pos = options.random_start_far_pos
        self.random_start_far_dis = options.random_start_far_dis
        self.do_plot = options.do_plot
        self.collision_bounce = options.collision_bounce

        self.enable_blocks = options.enable_blocks

    def random_set_start_pos(self, positions):
        self.start_pos = []
        self.start_angle = []
        if self.enable_blocks:
            # TODO:
            indexes = np.random.choice(range(0, len(positions)), 4, replace=False)
            if self.random_start_far_pos:
                while not self.if_start_positions_valid([positions[indexes[0]], positions[indexes[1]]],
                                                        [positions[indexes[2]], positions[indexes[3]]]):
                    indexes = np.random.choice(range(0, len(positions)), 4, replace=False)

            for i in indexes:
                self.start_pos.append(positions[i])
        else:
            positions_r = []
            positions_b = []
            for i in range(self.robot_r_num):
                positions_r.append([np.random.uniform(30, 778), np.random.uniform(30, 418)])
            for i in range(self.robot_b_num):
                positions_b.append([np.random.uniform(30, 778), np.random.uniform(30, 418)])
            while not self.if_start_positions_valid(positions_r, positions_b):  # 判断是否两个点集之间距离均远于300以及队友之间是否不重合
                positions_r = []
                positions_b = []
                for i in range(self.robot_r_num):
                    positions_r.append([np.random.uniform(30, 778), np.random.uniform(30, 418)])
                for i in range(self.robot_b_num):
                    positions_b.append([np.random.uniform(30, 778), np.random.uniform(30, 418)])
            self.start_pos = positions_r + positions_b
        for i in range(4):
            self.start_angle.append(np.random.random() * 360 - 180)

    def if_start_positions_valid(self, positions_r, positions_b):
        if len(positions_r) > 1:
            if np.linalg.norm(np.array(positions_r[0]) - np.array(positions_r[1])) < 60:
                return False
        if len(positions_b) > 1:
            if np.linalg.norm(np.array(positions_b[0]) - np.array(positions_b[1])) < 60:
                return False
        for pos_r in positions_r:
            for pos_b in positions_b:
                if np.linalg.norm(np.array(pos_r) - np.array(pos_b)) < self.random_start_far_dis:
                    return False
        return True


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

        # 手动调试内容：
        self.single_input = options.single_input
        # 每一帧的延迟
        self.delay_per_frame = options.time_delay_frame

    def init_render(self, options):
        from robomaster2D.envs.kernel_user_interface import User_Interface
        self.module_UI = User_Interface(
            self.state, self.module_engine, self.orders, self.map, options)
        self.module_engine.init_render()
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
        done = False
        if self.render_inited:
            self.step_feedback_UI()
        # 判断游戏是否暂停或等待用户输入指令
        if not self.state.pause and not self.state.wait_for_user_input:
            # 如果启动了就开始调用agent解码指令
            self.orders.combine(self.agents[0].decode_actions(self.state, actions[0]),
                                self.agents[1].decode_actions(self.state, actions[1]))
            # 动作输入为None时表示由键盘控制
            self.step_reset()
            for n in range(self.parameters.frame_num_one_step):
                if self.tick():  # 一个周期，5ms
                    done = True
            self.state.step += 1
            if self.single_input:
                self.state.wait_for_user_input = True
        if self.render_inited:
            self.render()
        # episode_step如果不为零，当step数量达到这个值，将提前结束episode
        if self.parameters.episode_step:
            if self.state.step == self.parameters.episode_step:
                done = True
        red_win = True
        blue_win = True
        for n in range(self.parameters.robot_r_num):
            if self.state.robots[n].hp > 0:
                blue_win = False
        for n in range(self.parameters.robot_b_num):
            if self.state.robots[n + self.parameters.robot_r_num].hp > 0:
                red_win = False
        if red_win:
            self.state.r_win_record.win()
        elif blue_win:
            self.state.r_win_record.fail()
        elif done:
            self.state.r_win_record.draw()
        # TODO: 对局结束时，双方机器人尚有存活的话，伤害高的一方获胜
        done = done or red_win or blue_win
        return done

    #
    # def step_orders(self, orders):
    #     self.step_feedback_UI()
    #     # 如果启动了就开始解码指令
    #     if not self.state.pause:
    #         self.orders.combine(*orders)
    #         self.step_reset()
    #     for n in range(self.parameters.frame_num_one_time):
    #         if self.tick():  # 一个周期，5ms
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
    ----tick函数----
    该函数运行游戏的一帧，返回是否done和是否pause的布尔值
    '''

    def tick(self):
        time.sleep(self.delay_per_frame)
        self.state.tick()  # 更新时间和buff等
        # 如果时间未到，则直接返回
        if self.state.if_not_started_yet():
            return False
        # 启动摄像头、雷达
        self.run_camera_lidar()
        # 运行引擎，计算一帧下机器人和子弹的运动
        act_feedback = self.module_engine.tick(self.orders)
        # 运行裁判系统：检测装甲板是否要扣血；冷却枪管；检测buff
        self.module_referee.tick(act_feedback)
        # 更新frame和robots的hp
        self.state.finish_tick()
        # 计时结束
        if self.state.if_end():
            # 此处不必将self.state.frame置零，因为self.state.reset函数会做这一项工作
            return True
        return False

    def render(self):
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

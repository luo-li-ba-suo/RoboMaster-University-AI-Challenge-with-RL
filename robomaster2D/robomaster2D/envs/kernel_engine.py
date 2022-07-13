from robomaster2D.envs.kernel_referee import *
from robomaster2D.envs.kernel_objects import Bullet
import ctypes


class Acts(object):  # 底层动作
    def __init__(self, rotate_speed=0, yaw_speed=0, shoot=0):
        self.rotate_speed = rotate_speed  # 底盘旋转速度
        self.yaw_speed = yaw_speed  # 云台旋转速度
        self.x_speed = 0  # 前进后退速度
        self.y_speed = 0  # 左右平移速度
        self.shoot = shoot  # 是否发射
        self.shoot_target_enemy = -1
        self.shoot_mutiple = False  # 是否连发
        self.dodge_dir = True
        self.dir_relate_to_map = True


class Act_feedback(object):
    def __init__(self, state):
        self.hit_source = [{'behind': [], 'front': [], 'left': [], 'right': [], 'not_armor': []},
                           {'behind': [], 'front': [], 'left': [], 'right': [], 'not_armor': []},
                           {'behind': [], 'front': [], 'left': [], 'right': [], 'not_armor': []},
                           {'behind': [], 'front': [], 'left': [], 'right': [], 'not_armor': []}]  # 存放碰撞类型

    #     self.hit_source_robot_collision = [{'behind': [], 'front': [], 'left': [], 'right': [], 'not_armor': []},
    #                                        {'behind': [], 'front': [], 'left': [], 'right': [], 'not_armor': []},
    #                                        {'behind': [], 'front': [], 'left': [], 'right': [], 'not_armor': []},
    #                                        {'behind': [], 'front': [], 'left': [], 'right': [],
    #                                         'not_armor': []}]  # 存放碰撞对象
    #
    # def robot_collision_to_source(self):  # 机器人碰撞信息可能有重复
    #     for n in range(self.state.robot_num):
    #         for part in ['behind', 'front', 'left', 'right', 'not_armor']:
    #             for hit in self.hit_source_robot_collision[n][part]:
    #                 self.hit_source[n][part].append('ROBOT')
    #                 self.hit_source[hit][part].append('ROBOT')

    def reset(self):
        self.hit_source = [{'behind': [], 'front': [], 'left': [], 'right': [], 'not_armor': []},
                           {'behind': [], 'front': [], 'left': [], 'right': [], 'not_armor': []},
                           {'behind': [], 'front': [], 'left': [], 'right': [], 'not_armor': []},
                           {'behind': [], 'front': [], 'left': [], 'right': [], 'not_armor': []}]  # 存放碰撞类型
        # self.hit_source_robot_collision = [{'behind': [], 'front': [], 'left': [], 'right': [], 'not_armor': []},
        #                                    {'behind': [], 'front': [], 'left': [], 'right': [], 'not_armor': []},
        #                                    {'behind': [], 'front': [], 'left': [], 'right': [], 'not_armor': []},
        #                                    {'behind': [], 'front': [], 'left': [], 'right': [],
        #                                     'not_armor': []}]  # 存放碰撞对象


class Route_Plan(object):
    def __init__(self, options):
        self.blocks = []  # 障碍物
        self.goals = {}
        self.kernel_astar = []
        self.robot_num = options.robot_r_num + options.robot_b_num
        ll = ctypes.cdll.LoadLibrary
        for n in range(self.robot_num):
            self.kernel_astar.append(ll("./build/libicra_planning_0.so"))
            self.kernel_astar[-1].init()
        buff_red = [True, True, False, False, True, True] if options.buff_mode else []
        buff_blue = [False, False, True, True, True, True] if options.buff_mode else []
        block_robot = [True] * self.robot_num
        self.block_info = []
        for n in range(options.robot_r_num):
            block_robot_copy = block_robot.copy()
            block_robot_copy[n] = False
            self.block_info.append(block_robot_copy + buff_red.copy())
        for n in range(options.robot_b_num):
            block_robot_copy = block_robot.copy()
            block_robot_copy[n + options.robot_r_num] = False
            self.block_info.append(block_robot_copy + buff_blue.copy())
        '''
        For example: four robots
                            rhp   rbu   bhp     bbu    ns    nm    r1     r2    b1    b2
        self.block_info = [[True, True, False, False, True, True, False, True, True, True],
                           [True, True, False, False, True, True, True, False, True, True],
                           [False, False, True, True, True, True, True, True, False, True],
                           [False, False, True, True, True, True, True, True, True, False]]
        '''
        self.is_navs = [False] * self.robot_num

    def show_blocks(self, agent_idx):
        self.kernel_astar[agent_idx].show_blocks()

    def reset_goal(self, goal, agent_idx):
        self.kernel_astar[agent_idx].set_goal(goal)
        self.goals[agent_idx] = goal

    def reset_block(self, blocks):
        for n in range(self.robot_num):
            self.kernel_astar[n].clean_obstacle()
            for n_block in range(len(self.block_info[n])):
                if self.block_info[n][n_block]:
                    self.kernel_astar[n].add_obstacle(int(blocks[n_block][0]),
                                                      int(blocks[n_block][1]),
                                                      int(blocks[n_block][2]),
                                                      int(blocks[n_block][3]),
                                                      int(blocks[n_block][4]),
                                                      int(blocks[n_block][5]),
                                                      int(blocks[n_block][6]),
                                                      int(blocks[n_block][7]))

    def update_plan(self, x, y, angle, robot_idx):
        self.kernel_astar[robot_idx].update_pos.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.kernel_astar[robot_idx].update_pos(
            ctypes.c_double(x),
            ctypes.c_double(y),
            ctypes.c_double(angle))
        self.kernel_astar[robot_idx].path_plan()
        self.kernel_astar[robot_idx].get_robot_vx.restype = ctypes.c_double
        self.kernel_astar[robot_idx].get_robot_vy.restype = ctypes.c_double
        self.kernel_astar[robot_idx].get_robot_angular.restype = ctypes.c_double
        vx = self.kernel_astar[robot_idx].get_robot_vx()
        vy = self.kernel_astar[robot_idx].get_robot_vy()
        vr = self.kernel_astar[robot_idx].get_robot_angular()
        is_Nav = self.kernel_astar[robot_idx].isNav()

        return vx, vy, vr, is_Nav


class Engine(object):
    def __init__(self, state, referee, options, map):
        self.acts = []
        self.act_feedback = []
        self.robot_r_num = options.robot_r_num
        self.robot_b_num = options.robot_b_num
        self.robot_num = options.robot_r_num + options.robot_b_num
        self.state = state
        self.impact_effect = options.impact_effect
        self.collision_bounce = options.collision_bounce
        self.unlimited_bullet = options.unlimited_bullet
        self.frame_num_one_second = options.frame_num_one_second
        self.map = map
        self.referee = referee
        self.render_inited = False
        if self.map.barriers.any():
            b = self.map.barriers[4]
            # 以下四个截距是中心正方形的四条边的截距，用作边缘碰撞检测
            self.intercept1 = (sum(b)) / 2 + (b[1] - b[0]) / 1.414214
            self.intercept2 = (sum(b)) / 2 - (b[1] - b[0]) / 1.414214
            self.intercept3 = (b[0] + b[1]) / 2 - (b[2] + b[3]) / 2 + (b[1] - b[0]) / 1.414214
            self.intercept4 = (b[0] + b[1]) / 2 - (b[3] + b[2]) / 2 - (b[1] - b[0]) / 1.414214

        self.impact_part_id2str = {0: 'behind', 1: 'front', 2: 'left', 3: 'right', 'not_armor': 'not armor'}

        for _ in range(self.robot_num):
            self.acts.append(Acts())
            self.act_feedback = Act_feedback(state)

        # route plan
        if options.do_route_plan:
            # information for route plan model
            self.route_plan = Route_Plan(options)
        else:
            self.route_plan = None

        '''自瞄'''
        self.theta = np.rad2deg(np.arctan(45 / 60))

    def init_render(self):
        self.orders_text = ['' for _ in range(self.robot_num)]
        self.points_for_render = [[] for _ in range(self.robot_num)]
        self.render_inited = True

    def tick(self, orders):
        # if not self.state.frame % 30 and self.route_plan:  # 150ms更新一次路径规划的障碍物分布
        #     self.route_plan.reset_block(self.state.blocks)
        self.orders_to_acts(orders)
        self.freeze_step()
        self.move_robot()
        if self.state.if_alarm20hz_goes_off():  # 每50ms开启射击开关(由於比賽規則規定裝甲板檢測受攻擊的是)
            self.shoot()
        i = 0
        while len(self.state.bullets):
            if self.move_check_bullet(i) or self.state.bullets[i].disappear():
                del self.state.bullets[i]  # 如果子弹超过射程、碰到墙壁 障碍 机器人 则消失
            else:
                i += 1
            if i >= len(self.state.bullets):
                break
        return self.act_feedback

    def orders_to_acts(self, orders):  # 将指令转化为底层动作
        # turn orders to acts
        self.orders_text = ['' for n in range(self.robot_num)]
        for n in range(self.robot_num):
            if self.render_inited:
                self.orders_text[n] += str(orders.set[n].x)
                self.orders_text[n] += str(orders.set[n].y)
                self.orders_text[n] += str(orders.set[n].rotate)
                self.orders_text[n] += str(orders.set[n].yaw)
                self.orders_text[n] += str(orders.set[n].shoot)
                self.orders_text[n] += str(orders.set[n].shoot_target_enemy)
            if self.state.robots[n].hp <= 0:
                continue
            if not orders.set[n].do_route_plan:  # 如果输入的是离散的移动指令
                # 加速：
                self.acts[n].x_speed += orders.set[n].x * self.state.robots[n].speed_acceleration
                self.acts[n].y_speed += orders.set[n].y * self.state.robots[n].speed_acceleration
                self.acts[n].rotate_speed += orders.set[n].rotate * self.state.robots[n].rotate_acceleration
                # 因阻力减速:
                if orders.set[n].x == 0:
                    if self.acts[n].x_speed > 0: self.acts[n].x_speed -= self.state.robots[n].drag_acceleration
                    if self.acts[n].x_speed < 0: self.acts[n].x_speed += self.state.robots[n].drag_acceleration
                    if abs(self.acts[n].x_speed) < self.state.robots[n].drag_acceleration:
                        self.acts[n].x_speed = 0
                if orders.set[n].y == 0:
                    if self.acts[n].y_speed > 0: self.acts[n].y_speed -= self.state.robots[n].drag_acceleration
                    if self.acts[n].y_speed < 0: self.acts[n].y_speed += self.state.robots[n].drag_acceleration
                    if abs(self.acts[n].y_speed) < self.state.robots[n].drag_acceleration:
                        self.acts[n].y_speed = 0
                if orders.set[n].rotate == 0:
                    if self.acts[n].rotate_speed > 0:
                        self.acts[n].rotate_speed -= self.state.robots[n].rotate_drag_acceleration
                    if self.acts[n].rotate_speed < 0:
                        self.acts[n].rotate_speed += self.state.robots[n].rotate_drag_acceleration
                    if abs(self.acts[n].rotate_speed) < self.state.robots[n].rotate_drag_acceleration:
                        self.acts[n].rotate_speed = 0
                # 限速：
                if self.acts[n].x_speed >= self.state.robots[n].speed_max:
                    self.acts[n].x_speed = self.state.robots[n].speed_max
                if self.acts[n].x_speed <= -self.state.robots[n].speed_max:
                    self.acts[n].x_speed = -self.state.robots[n].speed_max
                if self.acts[n].y_speed >= self.state.robots[n].speed_max:
                    self.acts[n].y_speed = self.state.robots[n].speed_max
                if self.acts[n].y_speed <= -self.state.robots[n].speed_max:
                    self.acts[n].y_speed = -self.state.robots[n].speed_max
                if self.acts[n].rotate_speed > self.state.robots[n].rotate_speed_max:
                    self.acts[n].rotate_speed = self.state.robots[n].rotate_speed_max
                if self.acts[n].rotate_speed < -self.state.robots[n].rotate_speed_max:
                    self.acts[n].rotate_speed = -self.state.robots[n].rotate_speed_max
                self.acts[n].dir_relate_to_map = False
            elif self.route_plan is not None:  # 如果使用路径规划
                if not self.state.frame % orders.set[n].freq_update_goal:
                    goal = [int(orders.set[n].x), int(orders.set[n].y)]
                    self.route_plan.reset_goal(goal, n)
                vx, vy, vr, is_Nav = self.route_plan.update_plan(self.state.robots[n].x,
                                                                 self.state.robots[n].y,
                                                                 self.state.robots[n].angle)

                self.acts[n].x_speed = vy
                self.acts[n].y_speed = vx
                self.acts[n].rotate_speed = vr * 0.005
                self.acts[n].dir_relate_to_map = orders[n].dir_relate_to_map
            else:
                print('Fail to transform orders to actions')

            # rotate yaw
            m = orders.set[n].shoot_target_enemy
            m = self.get_valid_target_index(n, m)
            if m is not None:
                self.acts[n].yaw_speed, enemy_aimed = self.auto_aim(n, m)
                if enemy_aimed:
                    self.state.robots[n].aimed_enemy = m
                else:
                    self.state.robots[n].aimed_enemy = None
            else:
                self.state.robots[n].aimed_enemy = None
                self.acts[n].yaw_speed = 0
                # # 加速：
                # if orders.set[n].yaw != 0:
                #     self.acts[n].yaw_speed += orders.set[n].yaw * self.state.robots[n].yaw_acceleration
                # # 因阻力减速
                # else:
                #     if self.acts[n].yaw_speed > 0: self.acts[n].yaw_speed -= self.state.robots[n].yaw_drag_acceleration
                #     if self.acts[n].yaw_speed < 0: self.acts[n].yaw_speed += self.state.robots[n].yaw_drag_acceleration
                #     if abs(self.acts[n].yaw_speed) < self.state.robots[n].yaw_drag_acceleration:
                #         self.acts[n].yaw_speed = 0
                # if self.acts[n].yaw_speed > self.state.robots[n].yaw_rotate_speed_max:
                #     self.acts[n].yaw_speed = self.state.robots[n].yaw_rotate_speed_max
                # if self.acts[n].yaw_speed < -self.state.robots[n].yaw_rotate_speed_max:
                #     self.acts[n].yaw_speed = -self.state.robots[n].yaw_rotate_speed_max
            self.acts[n].shoot = orders.set[n].shoot

    def move_robot(self):
        for n in range(self.robot_num):
            if self.state.robots[n].hp == 0:
                continue
            # move gimbal
            self.rotate_gimbal(n, self.acts[n].yaw_speed)
            # move chassis
            if self.state.robots[n].freeze_state[1] != 1:  # 禁止移动
                # rotate chassis
                if self.acts[n].rotate_speed:
                    p = self.state.robots[n].angle
                    self.state.robots[n].angle += self.acts[n].rotate_speed
                    if self.state.robots[n].angle > 180: self.state.robots[n].angle -= 360
                    if self.state.robots[n].angle < -180: self.state.robots[n].angle += 360
                    if self.impact_effect:
                        if self.check_interface(n):
                            if self.collision_bounce:
                                self.acts[n].rotate_speed *= -self.state.robots[n].move_discount
                            else:
                                self.acts[n].rotate_speed = 0
                            self.state.robots[n].angle = p

                # move x and y
                if self.acts[n].x_speed or self.acts[n].y_speed:
                    angle = np.deg2rad(self.state.robots[n].angle)
                    # x
                    p = self.state.robots[n].x
                    if not self.acts[n].dir_relate_to_map:
                        self.state.robots[n].vx = self.acts[n].x_speed * np.cos(angle) - self.acts[n].y_speed * np.sin(
                            angle)
                        self.state.robots[n].x += self.state.robots[n].vx
                    else:
                        self.state.robots[n].x += self.acts[n].x_speed
                    self.state.robots[n].center = np.array([self.state.robots[n].x, self.state.robots[n].y])
                    if self.impact_effect:
                        if self.check_interface(n):
                            if self.collision_bounce:
                                self.acts[n].x_speed *= -self.state.robots[n].move_discount
                            else:
                                self.acts[n].x_speed = 0
                            self.state.robots[n].x = p
                    # y
                    p = self.state.robots[n].y
                    if not self.acts[n].dir_relate_to_map:
                        self.state.robots[n].vy = self.acts[n].x_speed * np.sin(angle) + self.acts[n].y_speed * np.cos(
                            angle)
                        self.state.robots[n].y += self.state.robots[n].vy
                    else:
                        self.state.robots[n].y += self.acts[n].y_speed
                    self.state.robots[n].center = np.array([self.state.robots[n].x, self.state.robots[n].y])
                    if self.impact_effect:
                        if self.check_interface(n):
                            if self.collision_bounce:
                                self.acts[n].y_speed *= -self.state.robots[n].move_discount
                            else:
                                self.acts[n].y_speed = 0
                            self.state.robots[n].y = p
                    self.state.robots[n].center = np.array([self.state.robots[n].x, self.state.robots[n].y])
                else:
                    self.state.robots[n].vx = 0
                    self.state.robots[n].vy = 0
                if not self.impact_effect:
                    self.check_in_map(n)  # 用于取消了所有撞击效果时防止超出地图10单位

    def can_target_enemy_be_seen(self, n, m):
        if m in np.where((self.state.camera_vision[n] == 1))[0]:
            return True
        return False

    def get_valid_target_index(self, n, m):
        assert m >= 0, 'get_valid_target_index error'
        if n < self.state.robot_r_num:
            m += self.state.robot_r_num
        # 以下是敌人血量判断以及是否能观测到的判断，如果选中的敌人是死亡状态或不能观测，则不瞄准，自动换另一个敌人
        if self.state.robots[m].hp == 0 or not self.can_target_enemy_be_seen(n, m):
            if n < self.state.robot_r_num and self.state.robot_b_num > 1:
                if m == self.state.robot_r_num:
                    m += 1
                else:
                    m -= 1
            elif n >= self.state.robot_r_num > 1:
                if m == 0:
                    m += 1
                else:
                    m -= 1
            if self.state.robots[m].hp == 0 or not self.can_target_enemy_be_seen(n, m):
                return None
        return m

    def auto_aim(self, n, m):
        x = self.state.robots[m].x - self.state.robots[n].x
        y = self.state.robots[m].y - self.state.robots[n].y
        angle = np.angle(x + y * 1j, deg=True) - self.state.robots[m].angle
        if angle >= 180: angle -= 360
        if angle <= -180: angle += 360
        if -self.theta <= angle < self.theta:
            armor = get_armor_center(self.state.robots[m], 2)
        elif self.theta <= angle < 180 - self.theta:
            armor = get_armor_center(self.state.robots[m], 3)
        elif -180 + self.theta <= angle < -self.theta:
            armor = get_armor_center(self.state.robots[m], 1)
        else:
            armor = get_armor_center(self.state.robots[m], 0)
        x = armor[0] - self.state.robots[n].x
        y = armor[1] - self.state.robots[n].y

        angle = np.angle(x + y * 1j, deg=True) - self.state.robots[n].yaw - self.state.robots[n].angle
        if angle >= 180: angle -= 360
        if angle <= -180: angle += 360
        if angle > self.state.robots[n].yaw_rotate_speed_max:
            return self.state.robots[n].yaw_rotate_speed_max, False
        elif angle < -self.state.robots[n].yaw_rotate_speed_max:
            return -self.state.robots[n].yaw_rotate_speed_max, False
        else:
            return angle, True

    def rotate_gimbal(self, n, yaw_speed):
        if yaw_speed:
            self.state.robots[n].yaw += self.acts[n].yaw_speed
            if self.state.robots[n].yaw > 90: self.state.robots[n].yaw = 90
            if self.state.robots[n].yaw < -90: self.state.robots[n].yaw = -90

    def freeze_step(self):
        for n in range(self.robot_num):
            if self.state.robots[n].freeze_time[0] > 0:  # 禁止射击
                self.state.robots[n].freeze_time[0] -= 1
            else:
                self.state.robots[n].freeze_state[0] = 0
            if self.state.robots[n].freeze_time[1] > 0:  # 禁止移动
                self.state.robots[n].freeze_time[1] -= 1
            else:
                self.state.robots[n].freeze_state[1] = 0

    def shoot(self):
        for n in range(self.robot_num):
            if self.state.robots[n].hp <= 0:
                continue
            if not self.state.robots[n].freeze_state[0] and not self.state.robots[n].cannot_shoot_overheating:  # 禁止射击
                # fire or not
                if self.acts[n].shoot and (self.unlimited_bullet[n] or self.state.robots[n].bullet):
                    self.state.robots[n].bullet -= 1
                    self.state.bullets.append(
                        Bullet(self.state.robots[n].center, self.state.robots[n].yaw + self.state.robots[n].angle,
                               self.state.robots[n].bullet_speed, n))
                    self.state.robots[n].bullet_out_record.add()
                    self.state.robots[n].heat += self.state.robots[n].bullet_speed / \
                                                 (100 / self.frame_num_one_second)  # 100cm/m / 20frame/s

    def move_check_bullet(self, n):
        '''
                move bullet No.n, if interface with wall, barriers or robots, return True, else False
                if interface with robots, robots'hp will decrease
        '''
        previous_center = self.state.bullets[n].center.copy()
        self.state.bullets[n].center[0] += self.state.bullets[n].bullet_speed * np.cos(
            np.deg2rad(self.state.bullets[n].angle))
        self.state.bullets[n].center[1] += self.state.bullets[n].bullet_speed * np.sin(
            np.deg2rad(self.state.bullets[n].angle))
        return self.referee.check_bullet(self.state.bullets[n], previous_center, self.act_feedback)

    def check_interface(self, n):
        # robot barriers assess
        check_result = False
        armors, outlines = get_points_armor_vertex(self.state.robots[n])
        if self.render_inited:
            self.points_for_render[n] = armors + outlines
        for j in range(4):
            armor0 = armors[2 * j]
            armor1 = armors[2 * j + 1]
            outline0 = outlines[2 * j]
            outline1 = outlines[2 * j + 1]

            if (not 0 < outline0[0] < self.map.map_length or not 0 < outline0[1] < self.map.map_width) or \
                    (not 0 < outline1[0] < self.map.map_length or not 0 < outline1[1] < self.map.map_width):
                check_result = True
                if outline0[0] == outline1[0] or outline0[1] == outline1[1]:
                    self.act_feedback.hit_source[n][self.impact_part_id2str[j]].append('BOUNDARY')
                else:
                    self.act_feedback.hit_source[n]['not_armor'].append('BOUNDARY')

            outline_hit = self.referee.line_barriers_check(outline0, outline1)
            armor_hit = self.referee.line_barriers_check(armor0, armor1)

            if outline_hit:
                check_result = True
                if armor_hit:
                    self.act_feedback.hit_source[n][self.impact_part_id2str[j]].append('OBSTACLE')
                else:
                    self.act_feedback.hit_source[n]['not_armor'].append('OBSTACLE')
            else:
                if armor_hit:
                    self.act_feedback.hit_source[n][self.impact_part_id2str[j]].append('OBSTACLE')

        # robot robot assess
        for i in range(self.robot_num):
            if i == n: continue
            armors_tran = self.referee.transfer_to_robot_coordinate(armors, i)
            outlines_tran = self.referee.transfer_to_robot_coordinate(outlines, i)
            for j in range(4):
                armor0 = armors_tran[2 * j]
                armor1 = armors_tran[2 * j + 1]
                outline0 = outlines_tran[2 * j]
                outline1 = outlines_tran[2 * j + 1]
                hit_robot = (-22.5 <= outline0[0] <= 22.5 and -30 <= outline0[1] <= 30) or \
                            (-22.5 <= outline1[0] <= 22.5 and -30 <= outline1[1] <= 30)
                if not hit_robot:
                    hit_robot = self.referee.segment(outline0, outline1, [-22.5, -30], [22.5, 30]) or \
                                self.referee.segment(outline0, outline1, [-22.5, 30], [22.5, -30])
                hit_robot_at_armor = (-22.5 <= armor0[0] <= 22.5 and -30 <= armor0[1] <= 30) or \
                                     (-22.5 <= armor1[0] <= 22.5 and -30 <= armor1[1] <= 30)
                if not hit_robot_at_armor:
                    hit_robot_at_armor = self.referee.segment(armor0, armor1, [-22.5, -30], [22.5, 30]) or \
                                         self.referee.segment(armor0, armor1, [-22.5, 30], [22.5, -30])
                if hit_robot:
                    check_result = True
                    if hit_robot_at_armor:
                        self.act_feedback.hit_source[n][self.impact_part_id2str[j]].append('ROBOT')
                    else:
                        self.act_feedback.hit_source[n]['not_armor'].append('ROBOT')
        return check_result

    def check_in_map(self, n):
        if self.state.robots[n].x < -10:
            self.state.robots[n].x = -10
        if self.state.robots[n].y < -10:
            self.state.robots[n].y = -10
        if self.state.robots[n].x > self.map.map_length + 10:
            self.state.robots[n].x = self.map.map_length + 10
        if self.state.robots[n].y > self.map.map_width + 10:
            self.state.robots[n].y = self.map.map_width + 10

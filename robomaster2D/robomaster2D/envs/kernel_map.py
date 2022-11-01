import math

import numpy as np


def random_buff_info():  # 随机分配buff位置，第二列表示是否被使用
    tmp_buff_info = np.zeros((6, 2), dtype=np.int16)
    a = [0, 1, 2]
    b = [0, 1]
    c = [[0, 2], [1, 3], [4, 5]]
    np.random.shuffle(a)
    for ia, tmp_a in enumerate(a):
        np.random.shuffle(b)
        for ib, tmp_b in enumerate(b):
            tmp_buff_info[ia + 3 * ib] = (c[tmp_a][tmp_b], 1)
    return tmp_buff_info


class Map(object):
    def __init__(self, options):
        # 有关障碍物地图：
        self.obstacle_map = None
        self.obstacle_map_original = None
        self.obstacle_map_unit = options.obstacle_map_unit
        self.obstacle_map_size = options.obstacle_map_size
        self.robot_num = options.robot_r_num + options.robot_b_num
        self.death_idx = []

        self.map_length = 808
        self.map_width = 448
        self.start_areas = np.array([[[708.0, 808.0, 0.0, 100.0],  ###red start area
                                      [708.0, 808.0, 348.0, 448.0]],
                                     [[0.0, 100.0, 0.0, 100.0],  ###blue start area
                                      [0.0, 100.0, 348.0, 448.0]]], dtype='float32')
        self.buff_areas = np.array([[23.0, 77.0, 145.0, 193.0],
                                    [163.0, 217.0, 259.0, 307.0],
                                    [377.0, 431.0, 20.5, 68.5],
                                    [731.0, 785.0, 255.0, 303.0],
                                    [591.0, 645.0, 141.0, 189.0],
                                    [377.0, 431.0, 379.5, 427.5]], dtype='float32')
        self.barriers_expanded = np.array([[0.0, 110.0, 90.0, 130.0],
                                           [140.0, 240.0, 204.0, 244.0],
                                           [140.0, 180.0, 338.0, 448.0],
                                           [344.0, 464.0, 83.5, 123.5],
                                           [381.5, 426.5, 201.5, 246.5],
                                           [344.0, 464.0, 324.5, 364.5],
                                           [628.0, 668.0, 0.0, 110.0],
                                           [568.0, 668.0, 204.0, 244.0],
                                           [698.0, 808.0, 318.0, 358.0]], dtype='float32')
        if options.enable_blocks:
            self.barriers = np.array([[0.0, 100.0, 100.0, 120.0],
                                      [150.0, 230.0, 214.0, 234.0],
                                      [150.0, 170.0, 348.0, 448.0],
                                      [354.0, 454.0, 93.5, 113.5],
                                      [386.32, 421.67, 206.32, 241.68],
                                      [354.0, 454.0, 334.5, 354.5],
                                      [638.0, 658.0, 0.0, 100.0],
                                      [578.0, 658.0, 214.0, 234.0],
                                      [708.0, 808.0, 328.0, 348.0]], dtype='float32')
            # 障碍物类型。场地中有两种不同高度的障碍物，较矮的障碍物只能阻挡机器人前进不能阻挡子弹，标记为1,其他的标记为0
            self.barriers_mode = [0, 1, 0, 0, 1, 0, 0, 1, 0]
            self.barriers_pose = [0, 0, 0, 0, 1, 0, 0, 0, 0] # 0表示与坐标轴平行;1表示与坐标轴成45度角
        else:
            self.barriers = np.array([])
            self.barriers_mode = []
            self.barriers_pose = []
        self.goal_positions = [[50, 50], [100, 50], [150, 50], [200, 50], [250, 50], [300, 50],
                               [354, 50], [404, 50], [454, 50], [500, 50], [546, 50], [592, 50],
                               [708, 50], [758, 50],
                               [150, 110], [200, 110], [250, 110], [300, 110], [500, 110], [546, 110], [592, 110],
                               [708, 110], [758, 110],
                               [50, 170], [100, 170], [150, 170], [200, 170], [250, 170], [300, 170], [354, 170],
                               [404, 170], [454, 170], [500, 170], [550, 170], [600, 170],
                               [648, 170], [708, 170], [758, 170],
                               [50, 224], [100, 224], [280, 224], [330, 224], [478, 224], [528, 224], [708, 224],
                               [758, 224],
                               [50, 398], [100, 398], [216, 398], [262, 398], [308, 398], [354, 398], [404, 398],
                               [454, 398], [508, 398], [558, 398], [608, 398], [658, 398], [708, 398], [758, 398],
                               [658, 338], [608, 338], [558, 338], [508, 338], [308, 338], [262, 338], [216, 338],
                               [100, 338], [50, 338],
                               [758, 278], [708, 278], [658, 278], [608, 278], [558, 278], [508, 278], [454, 278],
                               [404, 278], [354, 278], [308, 278], [258, 278], [208, 278],
                               [160, 278], [100, 278], [50, 278]
                               ]
        self.max_dis = 1000

        # Astar
        self.Astar_obstacle_set_init = None
        self.Astar_obstacle_set_robots = []
        self.Astar_map_x_size = options.Astar_map_x_size
        self.Astar_map_y_size = options.Astar_map_y_size
        self.Astar_obstacle_expand = options.Astar_obstacle_expand
        self.bord = np.array([self.Astar_obstacle_expand, self.Astar_map_x_size-self.Astar_obstacle_expand,
                                                      self.Astar_obstacle_expand, self.Astar_map_y_size-self.Astar_obstacle_expand])

    def map_x_convert(self, pointx, expand=1):  # unit指网格化边长，expand指障碍物拓宽
        if pointx < expand:
            return 0
        elif pointx > self.map_length / self.obstacle_map_unit - expand:
            return self.map_length // self.obstacle_map_unit
        return pointx

    def map_y_convert(self, pointy, expand=1):
        if pointy < expand:
            return 0
        elif pointy > self.map_width / self.obstacle_map_unit - expand:
            return self.map_width // self.obstacle_map_unit
        return pointy

    def obstacle_map_init(self, expand=0):
        self.death_idx = []
        if self.obstacle_map_original is None:
            # 6个通道：两种地图障碍物占位/每个机器人占位
            self.obstacle_map = np.zeros((2 + self.robot_num,
                                          self.map_width // self.obstacle_map_unit + self.obstacle_map_size,
                                          self.map_length // self.obstacle_map_unit + self.obstacle_map_size), dtype=int)
            self.obstacle_map[0, :, 0:self.obstacle_map_size // 2] = 1
            self.obstacle_map[0, :, self.map_length // self.obstacle_map_unit + self.obstacle_map_size // 2:
                                    self.map_length // self.obstacle_map_unit + self.obstacle_map_size] = 1
            self.obstacle_map[0, 0:self.obstacle_map_size // 2, :] = 1
            self.obstacle_map[0, self.map_width // self.obstacle_map_unit + self.obstacle_map_size // 2:
                                 self.map_width // self.obstacle_map_unit + self.obstacle_map_size, :] = 1
            for i, barrier in enumerate(self.barriers):
                barrier = np.around(barrier / self.obstacle_map_unit)
                if i == 4:
                    p1, p2, p3, p4 = np.array([(barrier[0] + barrier[1]) / 2, barrier[2]]), \
                                     np.array([barrier[0], (barrier[2] + barrier[3]) / 2]), \
                                     np.array([barrier[1], (barrier[2] + barrier[3]) / 2]), \
                                     np.array([(barrier[0] + barrier[1]) / 2, barrier[3]])
                    self.fill_map_with_rectangle(1, np.array([p1, p2, p3, p4]), pre_clear=False)
                    continue
                xmin = int(self.map_x_convert(barrier[0], expand) + self.obstacle_map_size / 2 - expand)
                xmax = int(self.map_x_convert(barrier[1], expand) + self.obstacle_map_size / 2 + expand)
                ymin = int(self.map_y_convert(barrier[2], expand) + self.obstacle_map_size / 2 - expand)
                ymax = int(self.map_y_convert(barrier[3], expand) + self.obstacle_map_size / 2 + expand)
                if self.barriers_mode[i]:
                    self.obstacle_map[1, ymin:ymax, xmin:xmax] = 1
                else:
                    self.obstacle_map[0, ymin:ymax, xmin:xmax] = 1
            self.obstacle_map_original = self.obstacle_map.copy()
        else:
            self.obstacle_map = self.obstacle_map_original.copy()
        return self.obstacle_map

    def update_obstacle_map(self, robots):
        for i, robot in enumerate(robots):
            points = robot.outlines[[0, 2, 1, 3]]
            points[:, 0] = robot.outlines[[0, 2, 1, 3]][:, 0] * math.cos(math.radians(90-robot.angle)) + robot.outlines[[0, 2, 1, 3]][:,
                                                                                        1] * math.sin(math.radians(90-robot.angle))
            points[:, 1] = robot.outlines[[0, 2, 1, 3]][:, 1] * math.cos(math.radians(90-robot.angle)) - robot.outlines[[0, 2, 1, 3]][:,
                                                                                        0] * math.sin(math.radians(90-robot.angle))
            points += robot.center
            points = np.around(points / self.obstacle_map_unit)
            if robot.hp <= 0:
                if i not in self.death_idx:
                    # 如果机器人死亡，则算作固定障碍物;死亡只需添加一次
                    self.death_idx.append(i)
                    self.obstacle_map[i+2, ...] = 0
                    self.fill_map_with_rectangle(0, points, pre_clear=False)
            else:
                self.fill_map_with_rectangle(i + 2, points)
        for i, robot in enumerate(robots):
            if robot.friend is None:
                ids = [-2, -1, robot.id] + robot.enemy
            else:
                ids = [-2, -1, robot.id, robot.friend] + robot.enemy
            ids += np.array(2)
            x_range_0 = int(np.round(robot.x / self.obstacle_map_unit))
            # 误：x_range_1 = int(np.round(robot.x / self.obstacle_map_unit + self.obstacle_map_size))
            # 错误原因：np.round奇进偶舍
            x_range_1 = int(np.round(robot.x / self.obstacle_map_unit) + self.obstacle_map_size)
            y_range_0 = int(np.round(robot.y / self.obstacle_map_unit))
            y_range_1 = int(np.round(robot.y / self.obstacle_map_unit) + self.obstacle_map_size)

            robot.local_map = self.obstacle_map[ids, y_range_0:y_range_1, x_range_0:x_range_1]
        return self.obstacle_map

    def fill_map_with_rectangle(self, channel, points, pre_clear=True):
        if pre_clear:
            self.obstacle_map[channel, ...] = 0
        x_left = int(np.round(min(points[:, 0])))
        x_right = int(np.round(max(points[:, 0])))
        y_left = int(np.round(min(points[:, 1])))
        y_right = int(np.round(max(points[:, 1])))
        for x in range(x_left, x_right + 1):
            for y in range(y_left, y_right + 1):
                if (points[0][0] - x) * (points[1][1] - y) - (points[1][0] - x) * (points[0][1] - y) > 0:
                    continue
                elif (points[2][0] - x) * (points[0][1] - y) - (points[0][0] - x) * (points[2][1] - y) > 0:
                    continue
                elif (points[3][0] - x) * (points[2][1] - y) - (points[2][0] - x) * (points[3][1] - y) > 0:
                    continue
                elif (points[1][0] - x) * (points[3][1] - y) - (points[3][0] - x) * (points[1][1] - y) > 0:
                    continue
                self.obstacle_map[channel, int(y + self.obstacle_map_size / 2), int(x + self.obstacle_map_size / 2)] = 1

    def init_Astar_obstacle_set(self):
        if self.Astar_obstacle_set_init is None:
            self.Astar_obstacle_set_init = set()
            self.Add_Rec_Astar_obstacle_map(self.Astar_obstacle_set_init, self.bord)

            for i, barrier in enumerate(self.barriers):
                scale = self.Astar_map_x_size / self.map_length
                barrier = barrier.copy()
                barrier *= scale
                barrier = barrier.astype(np.int32)
                barrier += [-self.Astar_obstacle_expand, self.Astar_obstacle_expand,
                            -self.Astar_obstacle_expand, self.Astar_obstacle_expand]
                if self.barriers_pose[i] == 0:
                    self.Add_Rec_Astar_obstacle_map(self.Astar_obstacle_set_init, barrier, fill=True)
                else:
                    x_left = barrier[0]
                    x_right = barrier[1]
                    y_left = barrier[2]
                    y_right = barrier[3]
                    p1, p2, p3, p4 = np.array([(barrier[0] + barrier[1]) / 2, barrier[2]]), \
                                     np.array([barrier[0], (barrier[2] + barrier[3]) / 2]), \
                                     np.array([barrier[1], (barrier[2] + barrier[3]) / 2]), \
                                     np.array([(barrier[0] + barrier[1]) / 2, barrier[3]])
                    for x in range(x_left, x_right + 1):
                        for y in range(y_left, y_right + 1):
                            if (p1[0] - x) * (p2[1] - y) - (p2[0] - x) * (p1[1] - y) > 0:
                                continue
                            elif (p3[0] - x) * (p1[1] - y) - (p1[0] - x) * (p3[1] - y) > 0:
                                continue
                            elif (p4[0] - x) * (p3[1] - y) - (p3[0] - x) * (p4[1] - y) > 0:
                                continue
                            elif (p2[0] - x) * (p4[1] - y) - (p4[0] - x) * (p2[1] - y) > 0:
                                continue
                            self.Astar_obstacle_set_init.add((x, y))

        return self.Astar_obstacle_set_init

    def update_Astar_obstacle_set_robots(self, robots):
        self.Astar_obstacle_set_robots = []
        for robot in robots:
            obs = set()
            points = robot.outlines[0:4].copy()
            points_ = points + [[-15, -15], [15, -15],
                       [-15, 15], [15, 15]]
            points[:, 0] = points_[:, 0] * math.cos(math.radians(90 - robot.angle)) + \
                           points_[:, 1] * math.sin(math.radians(90 - robot.angle))
            points[:, 1] = points_[:, 1] * math.cos(math.radians(90 - robot.angle)) - \
                           points_[:, 0] * math.sin(math.radians(90 - robot.angle))
            points += robot.center
            scale = self.Astar_map_x_size / self.map_length
            points = np.around(points * scale)
            points = points.astype(np.int32)
            obs.add((points[0, 0], points[0, 1]))
            obs.add((points[1, 0], points[1, 1]))
            obs.add((points[2, 0], points[2, 1]))
            obs.add((points[3, 0], points[3, 1]))
            x_left = int(np.round(min(points[:, 0])))
            x_right = int(np.round(max(points[:, 0])))
            y_left = int(np.round(min(points[:, 1])))
            y_right = int(np.round(max(points[:, 1])))
            if points[0, 0] == points[1, 0] or points[0, 1] == points[1, 1]:  # 如果与坐标轴平行
                for x in range(x_left, x_right + 1):
                    for y in range(y_left, y_right + 1):
                        obs.add((x, y))
            else:
                mode0 = False
                if mode0:
                    obs_unit_last = None
                    for x in range(points[0, 0], points[1, 0] + 1) if points[0, 0] < points[1, 0] \
                            else range(points[1, 0], points[0, 0] + 1):
                        obs_unit = (x, self.calculate_coo_according_to_two_coo(points[0, 0], points[1, 0],
                                                                                points[0, 1], points[1, 1], x))
                        if obs_unit_last is not None:
                            if abs(obs_unit[1] - obs_unit_last[1]) > 1:
                                for y in range(obs_unit[1], obs_unit_last[1], 1 if obs_unit[1] < obs_unit_last[1] else -1):
                                    obs.add((obs_unit[0], y))
                        obs_unit_last = obs_unit
                        obs.add(obs_unit)
                    obs_unit_last = None
                    for x in range(points[0, 0], points[2, 0] + 1) if points[0, 0] < points[2, 0] else \
                                range(points[2, 0], points[0, 0] + 1):
                        obs_unit = (x, self.calculate_coo_according_to_two_coo(points[0, 0], points[2, 0],
                                                                        points[0, 1], points[2, 1], x))
                        if obs_unit_last is not None:
                            if abs(obs_unit[1] - obs_unit_last[1]) > 1:
                                for y in range(obs_unit[1], obs_unit_last[1], 1 if obs_unit[1] < obs_unit_last[1] else -1):
                                    obs.add((obs_unit[0], y))
                        obs_unit_last = obs_unit
                        obs.add(obs_unit)
                    obs_unit_last = None
                    for x in range(points[2, 0], points[3, 0] + 1) if points[2, 0] < points[3, 0] \
                            else range(points[3, 0], points[2, 0] + 1):
                        obs_unit = (x, self.calculate_coo_according_to_two_coo(points[2, 0], points[3, 0],
                                                                        points[2, 1], points[3, 1], x))
                        if obs_unit_last is not None:
                            if abs(obs_unit[1] - obs_unit_last[1]) > 1:
                                for y in range(obs_unit[1], obs_unit_last[1], 1 if obs_unit[1] < obs_unit_last[1] else -1):
                                    obs.add((obs_unit[0], y))
                        obs_unit_last = obs_unit
                        obs.add(obs_unit)
                    obs_unit_last = None
                    for x in range(points[1, 0], points[3, 0]+1) if points[1, 0] < points[3, 0] \
                            else range(points[3, 0], points[1, 0]+1):
                        obs_unit = (x, self.calculate_coo_according_to_two_coo(points[1, 0], points[3, 0],
                                                                        points[1, 1], points[3, 1], x))
                        if obs_unit_last is not None:
                            if abs(obs_unit[1] - obs_unit_last[1]) > 1:
                                for y in range(obs_unit[1], obs_unit_last[1], 1 if obs_unit[1] < obs_unit_last[1] else -1):
                                    obs.add((obs_unit[0], y))
                        obs_unit_last = obs_unit
                        obs.add(obs_unit)
                else:
                    for x in range(x_left, x_right + 1):
                        for y in range(y_left, y_right + 1):
                            if (points[0][0] - x) * (points[1][1] - y) - (points[1][0] - x) * (points[0][1] - y) < 0:
                                continue
                            elif (points[2][0] - x) * (points[0][1] - y) - (points[0][0] - x) * (points[2][1] - y) < 0:
                                continue
                            elif (points[3][0] - x) * (points[2][1] - y) - (points[2][0] - x) * (points[3][1] - y) < 0:
                                continue
                            elif (points[1][0] - x) * (points[3][1] - y) - (points[3][0] - x) * (points[1][1] - y) < 0:
                                continue
                            obs.add((x,y))
            self.Astar_obstacle_set_robots.append(obs)

    @staticmethod
    def calculate_coo_according_to_two_coo(x1, x2, y1, y2, x3):
        assert x1 != x2, "calculate_coo_according_to_two_coo error"
        return int(np.around((y1*(x2-x3)+y2*(x3-x1))/(x2-x1)))

    def get_Astar_obstacle_set(self, robot_idx):
        assert self.Astar_obstacle_set_init is not None, "Astar_obstacle set Not Initialised"
        assert self.Astar_obstacle_set_robots is not None, "Astar robots obstacle set Not updated"
        obs_set = self.Astar_obstacle_set_init.copy()
        for i in range(self.robot_num):
            if i != robot_idx:
                obs_set = obs_set.union(self.Astar_obstacle_set_robots[i])
        return obs_set

    @staticmethod
    def Add_Rec_Astar_obstacle_map(obs_map, rec, fill=False):
        if fill:
            for x in range(rec[0], rec[1]):
                for y in range(rec[2], rec[3]):
                    obs_map.add((x, y))
        else:
            for i in range(rec[0], rec[1]):
                obs_map.add((i, rec[2]))
            for i in range(rec[0], rec[1]):
                obs_map.add((i, rec[3] - 1))

            for i in range(rec[2], rec[3]):
                obs_map.add((rec[0], i))
            for i in range(rec[2], rec[3]):
                obs_map.add((rec[1] - 1, i))


if __name__ == "__main__":
    from robomaster2D.envs.options import Parameters
    from robomaster2D.envs.kernel_objects import Robot
    from robomaster2D.envs.src import plotting
    from robomaster2D.envs.kernel_Astar import AStar, search
    import time

    options = Parameters()
    map = Map(options)
    bm = map.obstacle_map_init()
    robots = [Robot(2, 2, x=70, y=90, angle=45),
              Robot(2, 2, x=540, y=100, angle=0),
              Robot(2, 2, x=310, y=200, angle=-45),
              Robot(2, 2, x=550, y=290, angle=90)]
    bm = map.update_obstacle_map(robots)

    s_start = tuple((robots[0].center * map.Astar_map_x_size/808).astype(int))
    s_goal = (54, 45-10)
    obs_set = map.init_Astar_obstacle_set()
    map.update_Astar_obstacle_set_robots(robots)
    obs_set = map.get_Astar_obstacle_set(0)
    plot = plotting.Plotting(s_start, s_goal, obs_set)

    start_time = time.time()
    path, visited = search(s_goal, obs_set, s_start, map.bord)
    print("cost time: ", time.time() - start_time)
    # while not path:
    #     s_goal
    plot.animation(path, visited, "A*")  # animation

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
        self.unit = options.local_map_unit
        self.part_map_size = options.local_map_size
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
        else:
            self.barriers = np.array([])
            self.barriers_mode = []
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

    def map_x_convert(self, pointx, expand=1):  # unit指网格化边长，expand指障碍物拓宽
        if pointx < expand:
            return 0
        elif pointx > self.map_length / self.unit - expand:
            return self.map_length // self.unit
        return pointx

    def map_y_convert(self, pointy, expand=1):
        if pointy < expand:
            return 0
        elif pointy > self.map_width / self.unit - expand:
            return self.map_width // self.unit
        return pointy

    def state_map_init(self, expand=3):
        self.state_map = np.ones((self.map_width // self.unit + self.part_map_size,
                                  self.map_length // self.unit + self.part_map_size), dtype=int)
        self.state_map[:, 0:self.part_map_size // 2] = 0
        self.state_map[:, self.map_length // self.unit + self.part_map_size // 2:
                          self.map_length // self.unit + self.part_map_size] = 0
        self.state_map[0:self.part_map_size // 2, :] = 0
        self.state_map[self.map_width // self.unit + self.part_map_size // 2:
                       self.map_width // self.unit + self.part_map_size, :] = 0
        for barrier in self.barriers:
            barrier = np.around(barrier / self.unit)
            xmin = int(self.map_x_convert(barrier[0], expand) + self.part_map_size / 2 - expand)
            xmax = int(self.map_x_convert(barrier[1], expand) + self.part_map_size / 2 + expand)
            ymin = int(self.map_y_convert(barrier[2], expand) + self.part_map_size / 2 - expand)
            ymax = int(self.map_y_convert(barrier[3], expand) + self.part_map_size / 2 + expand)
            self.state_map[ymin:ymax, xmin:xmax] = 0
        return self.state_map

    def update_state_map(self):
        self.state_map = self.state_map
        return self.state_map

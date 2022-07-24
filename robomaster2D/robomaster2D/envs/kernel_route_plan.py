"""
HITSZ RMUA 路径规划模块python接口
kideng
开发须知：
1.路径规划模块不能多次实例化，因为多次实例化时每个实例之间同步障碍物时会保持一致。
所以应用同一个路径规划实例来对每个智能体分别走完整套流程：
- 执行目标点赋予
- 障碍物同步
- 位置同步
- 速度获取
"""

import ctypes
import os

import numpy as np


class Route_Plan(object):
    def __init__(self, options):
        self.blocks = []  # 障碍物
        self.goals = {}
        self.robot_num = options.robot_r_num + options.robot_b_num
        ll = ctypes.cdll.LoadLibrary
        self.kernel_astar = ll(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            f"../path_planning/cmake-build-debug/libicra_planning.so"))
        self.kernel_astar.init(False)

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
        agents_need_map = ['src.agents.rl_trainer', 'src.agents.nn_enemy']
        self.robot_ids_need_map = []
        if options.red_agents_path in agents_need_map:
            for n in range(options.robot_r_num):
                self.robot_ids_need_map.append(n)
        if options.blue_agents_path in agents_need_map:
            for n in range(options.robot_b_num):
                self.robot_ids_need_map.append(n + options.robot_r_num)
        self.local_map_size = options.local_map_size

    def reset_goal(self, goal, robot_idx):
        self.goals[robot_idx] = goal

    def get_obstacle_map(self, robot_idx, center):
        # if robot_idx not in self.robot_ids_need_map:
        #     return None
        self.kernel_astar.clean_obstacle()
        if self.blocks is not None:
            for n_block in range(len(self.block_info[robot_idx])):
                if self.block_info[robot_idx][n_block]:
                    self.kernel_astar.add_obstacle(int(self.blocks[n_block][0]),
                                                   int(self.blocks[n_block][1]),
                                                   int(self.blocks[n_block][2]),
                                                   int(self.blocks[n_block][3]),
                                                   int(self.blocks[n_block][4]),
                                                   int(self.blocks[n_block][5]),
                                                   int(self.blocks[n_block][6]),
                                                   int(self.blocks[n_block][7]))
        robot_x = int(center[0] / 808.0 * 81)
        robot_y = int(center[1] / 448.0 * 45)

        f = self.kernel_astar.get_block_map
        f.restype = ctypes.POINTER((ctypes.c_int * 81) * 45)
        mat = []
        for i in f().contents:
            mat.append([])
            for j in i:
                j = 0 if j > 240 else 1
                mat[-1].append(j)
        global_map = np.ones((1, 45 + self.local_map_size // 2 * 2, 81 + self.local_map_size // 2 * 2))
        global_map[0, self.local_map_size // 2:45 + self.local_map_size // 2,
        self.local_map_size // 2:81 + self.local_map_size // 2] = mat
        local_map = global_map[:, robot_y:robot_y + self.local_map_size, robot_x:robot_x + self.local_map_size]
        return local_map

    def update_blocks(self, blocks):
        self.blocks = blocks

    def update_plan(self, x, y, angle, robot_idx, get_map=False):
        success = False
        vx = vy = vr = is_Nav = local_map = None
        while not success:
            try:
                self.kernel_astar.set_goal(self.goals[robot_idx][0], self.goals[robot_idx][1])
                self.kernel_astar.clean_obstacle()
                if self.blocks is not None:
                    for n_block in range(len(self.block_info[robot_idx])):
                        if self.block_info[robot_idx][n_block]:
                            self.kernel_astar.add_obstacle(int(self.blocks[n_block][0]),
                                                           int(self.blocks[n_block][1]),
                                                           int(self.blocks[n_block][2]),
                                                           int(self.blocks[n_block][3]),
                                                           int(self.blocks[n_block][4]),
                                                           int(self.blocks[n_block][5]),
                                                           int(self.blocks[n_block][6]),
                                                           int(self.blocks[n_block][7]))
                self.kernel_astar.update_pos.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
                self.kernel_astar.update_pos(
                    ctypes.c_double(x),
                    ctypes.c_double(y),
                    ctypes.c_double(angle))
                self.kernel_astar.path_plan()
                self.kernel_astar.get_robot_vx.restype = ctypes.c_double
                self.kernel_astar.get_robot_vy.restype = ctypes.c_double
                self.kernel_astar.get_robot_angular.restype = ctypes.c_double
                vx = self.kernel_astar.get_robot_vx() * 200
                vy = self.kernel_astar.get_robot_vy() * 200
                vr = self.kernel_astar.get_robot_angular() * 0.05

                is_Nav = self.kernel_astar.isNav()

                if get_map:
                    robot_x = int(x / 808.0 * 81)
                    robot_y = int(y / 448.0 * 45)

                    f = self.kernel_astar.get_block_map
                    f.restype = ctypes.POINTER((ctypes.c_int * 81) * 45)
                    mat = []
                    for i in f().contents:
                        mat.append([])
                        for j in i:
                            j = 0 if j > 240 else 1
                            mat[-1].append(j)
                    global_map = np.ones((45 + self.local_map_size // 2 * 2, 81 + self.local_map_size // 2 * 2))
                    global_map[self.local_map_size // 2:45 + self.local_map_size // 2,
                               self.local_map_size // 2:81 + self.local_map_size // 2] = mat
                    local_map = global_map[robot_y:robot_y + self.local_map_size, robot_x:robot_x + self.local_map_size]
                else:
                    local_map = None
                success = True
            except BaseException as e:
                print("RoutePlanningRuined")
                print(e)
                ll = ctypes.cdll.LoadLibrary
                self.kernel_astar = ll(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                    f"../path_planning/cmake-build-debug/libicra_planning.so"))
        return vx, vy, vr, is_Nav, local_map


if __name__ == "__main__":
    from robomaster2D.envs.options import Parameters

    args = Parameters()
    args.red_agents_path = 'src.agents.human_agent'
    args.blue_agents_path = 'src.agents.handcrafted_enemy'
    args.time_delay_frame = 0
    astar = Route_Plan(args)
    while True:
        astar.reset_goal([300, 300], 2)
        astar.update_plan(50, 50, 0, 2, None)

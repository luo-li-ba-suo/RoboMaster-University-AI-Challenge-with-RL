import ctypes
import os


class Route_Plan(object):
    def __init__(self, options):
        self.blocks = []  # 障碍物
        self.goals = {}
        self.kernel_astar = {}
        self.robot_num = options.robot_r_num + options.robot_b_num
        self.robot_ids = []
        ll = ctypes.cdll.LoadLibrary
        if options.red_agents_path == 'src.agents.handcrafted_enemy':
            for i in range(options.robot_r_num):
                self.robot_ids.append(i)
                self.kernel_astar[i] = ll(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       f"../path_planning/cmake-build-debug/libicra_planning.so"))
                self.kernel_astar[i].init(False)
        if options.blue_agents_path == 'src.agents.handcrafted_enemy':
            for i in range(options.robot_b_num):
                self.robot_ids.append(i + options.robot_r_num)
                self.kernel_astar[i + options.robot_r_num] = \
                    ll(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    f"../path_planning/cmake-build-debug/libicra_planning.so"))
                self.kernel_astar[i + options.robot_r_num].init(False)
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

        self.frame = 0

    def show_blocks(self, agent_idx):
        self.kernel_astar[agent_idx].show_blocks()

    def reset_goal(self, goal, robot_idx):
        self.goals[robot_idx] = goal

    def update_plan(self, x, y, angle, robot_idx, blocks=None, re_plan=True):
        success = False
        vx = vy = vr = is_Nav = None
        while not success:
            try:
                self.kernel_astar[robot_idx].clean_obstacle()
                self.kernel_astar[robot_idx].set_goal(self.goals[robot_idx][0], self.goals[robot_idx][1])
                if blocks is not None:
                    for n_block in range(len(self.block_info[robot_idx])):
                        if self.block_info[robot_idx][n_block]:
                            self.kernel_astar[robot_idx].add_obstacle(int(blocks[n_block][0]),
                                                                      int(blocks[n_block][1]),
                                                                      int(blocks[n_block][2]),
                                                                      int(blocks[n_block][3]),
                                                                      int(blocks[n_block][4]),
                                                                      int(blocks[n_block][5]),
                                                                      int(blocks[n_block][6]),
                                                                      int(blocks[n_block][7]))
                self.kernel_astar[robot_idx].update_pos.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
                self.kernel_astar[robot_idx].update_pos(
                    ctypes.c_double(x),
                    ctypes.c_double(y),
                    ctypes.c_double(angle))
                if re_plan:
                    self.kernel_astar[robot_idx].path_plan()
                self.kernel_astar[robot_idx].get_robot_vx.restype = ctypes.c_double
                self.kernel_astar[robot_idx].get_robot_vy.restype = ctypes.c_double
                self.kernel_astar[robot_idx].get_robot_angular.restype = ctypes.c_double
                vx = self.kernel_astar[robot_idx].get_robot_vx() * 20
                vy = self.kernel_astar[robot_idx].get_robot_vy() * 20
                vr = self.kernel_astar[robot_idx].get_robot_angular() * 0.05

                is_Nav = self.kernel_astar[robot_idx].isNav()
                success = True
            except BaseException as e:
                print("RoutePlanningRuined")
                print(e)
                ll = ctypes.cdll.LoadLibrary
                self.kernel_astar[robot_idx] = ll(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       f"../path_planning/cmake-build-debug/libicra_planning.so"))
        return vx, vy, vr, is_Nav


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

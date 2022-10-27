from robomaster2D.envs.src.agents_base import *
from robomaster2D.envs.kernel_Astar import search
import numpy as np
from robomaster2D.envs.src import plotting


class My_Agent(Base_Agent):
    def __init__(self, _id, options):
        super().__init__(_id, options)
        self.name = 'handcrafted_enemy'
        self.frame = 0
        self.path = [[] for _ in range(self.num_robots)]
        self.last_positions = [None for _ in range(self.num_robots)]
        self.path_cache_len = 10

    def reset(self):
        self.path = [[] for _ in range(self.num_robots)]
        self.last_positions = [None for _ in range(self.num_robots)]

    def decode_actions(self, game_state, actions=None):  # 根据动作编码，解码产生动作
        self.orders.reset()
        # if not self.frame%10:
        #     self.orders.reset()
        # if self.frame == 100:
        #     self.frame = 0
        #     for i in range(self.num_robots):
        #         self.orders.set[i].x = random.randint(-1, 1)
        #         self.orders.set[i].y = random.randint(-1, 1)
        #         self.orders.set[i].rotate = random.randint(-1, 1)
        #         self.orders.set[i].yaw = random.randint(-1, 1)
        #         self.orders.set[i].shoot_target_enemy = random.randint(-1, self.num_enemy_robots-1)
        # self.frame += 1
        """
        for i in range(self.num_robots):
            self.orders.set[i].do_route_plan = True
            if self.enemy_num > 1:
                target_enemy = self.enemy_ids[i]
                if game_state.robots[target_enemy].hp <= 0:
                    if i == 0:
                        target_enemy = self.enemy_ids[1]
                    else:
                        target_enemy = self.enemy_ids[0]
            else:
                target_enemy = self.enemy_ids[0]
            if game_state.camera_vision[self.robot_ids[i], target_enemy] and \
                game_state.dist_matrix[self.robot_ids[i]][target_enemy] < 250:
                self.orders.set[i].stop = True
            else:
                self.orders.set[i].stop = False
            self.orders.set[i].x = game_state.robots[target_enemy].x
            self.orders.set[i].y = game_state.robots[target_enemy].y
            if game_state.robots[self.robot_ids[i]].aimed_enemy is not None:
                self.orders.set[i].shoot = 1
            self.orders.set[i].dir_relate_to_map = True
        """
        '''用Astar算法计算路径前进方向'''
        game_state.map.init_Astar_obstacle_set()
        game_state.map.update_Astar_obstacle_set_robots(game_state.robots)
        for i, robot_id in enumerate(self.robot_ids):
            if self.enemy_num > 1:
                target_enemy = self.enemy_ids[i]
                self.orders.set[i].shoot_target_enemy = i
                if game_state.robots[target_enemy].hp <= 0:
                    if i == 0:
                        target_enemy = self.enemy_ids[1]
                        self.orders.set[i].shoot_target_enemy = 1
                    else:
                        target_enemy = self.enemy_ids[0]
                        self.orders.set[i].shoot_target_enemy = 0
            else:
                target_enemy = self.enemy_ids[0]
            if tuple(game_state.robots[robot_id].center) == self.last_positions[i]:
                self.orders.set[i].x = np.random.randint(low=-1, high=2)
                self.orders.set[i].y = np.random.randint(low=-1, high=2)
                self.orders.set[i].rotate = np.random.randint(low=-1, high=2)
            else:

                if game_state.camera_vision[robot_id, target_enemy] == 0 or \
                        game_state.dist_matrix[robot_id, target_enemy] > 200:  # TODO:300为子弹射程
                    start = (game_state.robots[robot_id].center * game_state.map.Astar_map_x_size / 808).astype(np.int32)
                    goal = (game_state.robots[target_enemy].center * game_state.map.Astar_map_x_size / 808).astype(np.int32)
                    if not self.path[i] or len(self.path[i]) < 2:
                        obs_set = game_state.map.get_Astar_obstacle_set(robot_id)
                        self.path[i], visited = search(s_start=tuple(start), goal=tuple(goal), obs=obs_set,
                                                       bord=game_state.map.bord)
                        plot = plotting.Plotting(tuple(start), tuple(goal), obs_set)
                        # plot.animation(self.path[i], visited, "A*", pause=False)
                    if self.path[i] and len(self.path[i]) > self.path_cache_len:
                        self.path[i] = self.path[i][-self.path_cache_len:]
                    if self.path[i]:
                        self.orders.set[i].x = self.path[i][-2][0] - self.path[i][-1][0]
                        self.orders.set[i].y = self.path[i][-2][1] - self.path[i][-1][1]
                        self.orders.set[i].rotate = np.angle(self.orders.set[i].x + self.orders.set[i].y*1j, deg=True)
                        self.orders.set[i].rotate_target_mode = True
                        del self.path[i][-1]
                else:
                    if self.path[i]:
                        del self.path[i][-1]
            self.orders.set[i].dir_relate_to_map = True
            self.orders.set[i].auto_rotate = True
            self.orders.set[i].shoot = 1 if game_state.robots[robot_id].aimed_enemy is not None else 0
            self.last_positions[i] = tuple(game_state.robots[robot_id].center)
        return self.orders

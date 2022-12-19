from robomaster2D.envs.src.agents_base import *
from robomaster2D.envs.kernel_Astar import search
import numpy as np
from robomaster2D.envs.src import plotting




class My_Agent(Base_Agent):
    def __init__(self, _id, options):
        super().__init__(_id, options)
        self.name = 'retreat_enemy'
        self.frame = 0
        self.path = [[] for _ in range(self.num_robots)]
        self.last_positions = [None for _ in range(self.num_robots)]
        self.last_angles = [None for _ in range(self.num_robots)]
        self.path_cache_len = 10

        self.need_new_goals = True
        self.last_goals = None

        self.enable_blocks = options.enable_blocks

        self.candidate_coordinate = np.array([[50, 50], [708, 50], [100, 398], [758, 398], [404, 50], [404, 398]])

    def reset(self):
        self.path = [[] for _ in range(self.num_robots)]
        self.last_positions = [None for _ in range(self.num_robots)]
        self.last_angles = [None for _ in range(self.num_robots)]
        self.need_new_goals = True

    def decode_actions(self, game_state, actions=None):  # 根据动作编码，解码产生动作
        self.orders.reset()

        assert self.robot_red_num == 2 and self.robot_blue_num == 2, \
            f"{self.robot_red_num} red and {self.robot_blue_num} blue not implemented"

        '''判断撤退位置'''
        if self.need_new_goals:
            # self.need_new_goals = False
            self.last_goals = self.get_new_goals(game_state)
        goals = self.last_goals
        '''用Astar算法计算路径前进方向'''
        game_state.map.init_Astar_obstacle_set()
        game_state.map.update_Astar_obstacle_set_robots(game_state.robots)
        for i, robot_id in enumerate(self.robot_ids):
            if game_state.robots[robot_id].hp <= 0:
                continue
            if tuple(game_state.robots[robot_id].center) == self.last_positions[i]:
                if not self.if_arrive(game_state.robots[robot_id].center, goals[i]):
                    self.orders.set[i].x = np.random.randint(low=-1, high=2)
                    self.orders.set[i].y = np.random.randint(low=-1, high=2)
                    self.orders.set[i].rotate = np.random.randint(low=-1, high=2)
            else:
                start = (game_state.robots[robot_id].center * game_state.map.Astar_map_x_size / 808).astype(np.int32)
                goal = (goals[i] * game_state.map.Astar_map_x_size / 808).astype(np.int32)
                if not self.path[i] or len(self.path[i]) < 2:
                    obs_set = game_state.map.get_Astar_obstacle_set(robot_id)
                    self.path[i], visited = search(s_start=tuple(goal), goal=tuple(start), obs=obs_set,
                                                   bord=game_state.map.bord)
                    plot = plotting.Plotting(tuple(goal), tuple(start), obs_set)
                    # plot.animation(self.path[i], visited, "A*", pause=False)
                if self.path[i] and len(self.path[i]) > self.path_cache_len:
                    self.path[i] = self.path[i][0:self.path_cache_len]
                if self.path[i]:
                    self.orders.set[i].x = self.path[i][1][0] - self.path[i][0][0]
                    self.orders.set[i].y = self.path[i][1][1] - self.path[i][0][1]
                    self.orders.set[i].rotate = np.angle(self.orders.set[i].x + self.orders.set[i].y*1j, deg=True)
                    self.orders.set[i].rotate_target_mode = True
                    del self.path[i][-1]
            self.orders.set[i].move_along_the_axis = True
            self.orders.set[i].auto_rotate = True
            self.orders.set[i].shoot_target_enemy = i
            self.orders.set[i].shoot = 1 if game_state.robots[robot_id].aimed_enemy is not None else 0
            self.last_positions[i] = tuple(game_state.robots[robot_id].center)
            self.last_angles[i] = game_state.robots[robot_id].angle
        return self.orders

    def if_safe(self, point, safe_dis=40):
        for safe_point in self.candidate_coordinate[:4]:
            if np.linalg.norm(point-safe_point) < safe_dis:
                return True
        return False

    def if_arrive(self, point, goal):
        if np.linalg.norm(np.array(point)-goal) < 10:
            return True
        return False

    def assign_retreat_coordinates(self, game_state, retreat_points, robot_centers):
        robot_0_dis = [np.linalg.norm(robot_centers[0] - retreat_point) for retreat_point in retreat_points]
        robot_1_dis = [np.linalg.norm(robot_centers[1] - retreat_point) for retreat_point in retreat_points]
        if game_state.robots[self.robot_ids[0]].hp <= 0:
            if np.argmin(robot_1_dis) == 0:
                return 1, 0
            else:
                return 0, 1
        if game_state.robots[self.robot_ids[1]].hp <= 0:
            if np.argmin(robot_0_dis) == 0:
                return 0, 1
            else:
                return 1, 0

        if np.min(robot_0_dis) < np.min(robot_1_dis):
            if np.argmin(robot_0_dis) == 0:
                return 0, 1
            else:
                return 1, 0
        else:
            if np.argmin(robot_1_dis) == 0:
                return 1, 0
            else:
                return 0, 1

    def get_new_goals(self, game_state):
        """判断撤退位置"""
        enemy_coordinates = np.array([game_state.robots[enemy].center for enemy in self.enemy_ids])
        our_coordinates = np.array([game_state.robots[i].center for i in self.robot_ids])
        if (enemy_coordinates[:, 0] < 404).all():
            candidates = [1, 3]
            goal_0, goal_1 = self.assign_retreat_coordinates(game_state, self.candidate_coordinate[candidates],
                                                             our_coordinates)
        elif (enemy_coordinates[:, 0] >= 404).all():
            candidates = [0, 2]
            goal_0, goal_1 = self.assign_retreat_coordinates(game_state, self.candidate_coordinate[candidates],
                                                             our_coordinates)
        elif (enemy_coordinates[:, 1] < 224).all():
            candidates = [3, 5]
            goal_0, goal_1 = self.assign_retreat_coordinates(game_state, self.candidate_coordinate[candidates],
                                                             our_coordinates)
        elif (enemy_coordinates[:, 1] >= 224).all():
            candidates = [0, 4]
            goal_0, goal_1 = self.assign_retreat_coordinates(game_state, self.candidate_coordinate[candidates],
                                                             our_coordinates)
        elif ((enemy_coordinates[:, 1]) >= 224 & (enemy_coordinates[:, 0] >= 404)).any():
            candidates = [1, 2]
            goal_0, goal_1 = self.assign_retreat_coordinates(game_state, self.candidate_coordinate[candidates],
                                                             our_coordinates)
        else:
            candidates = [0, 3]
            goal_0, goal_1 = self.assign_retreat_coordinates(game_state, self.candidate_coordinate[candidates],
                                                             our_coordinates)
        return [self.candidate_coordinate[candidates[goal_0]], self.candidate_coordinate[candidates[goal_1]]]

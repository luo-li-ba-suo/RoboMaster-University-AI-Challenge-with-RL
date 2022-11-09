import copy


class Base_Agent(object):
    def __init__(self, _id, options):
        self.id = _id
        self.robot_blue_num = options.robot_b_num
        self.robot_red_num = options.robot_r_num
        self.num_robots = options.robot_b_num if _id else options.robot_r_num
        self.enemy_num = options.robot_r_num if _id else options.robot_b_num
        # robot id 表示agent对应的robot在kernel_game中的索引
        self.robot_ids = [(i + self.robot_red_num if _id else i) for i in range(self.num_robots)]
        self.enemy_ids = [(i if _id else i + self.robot_red_num) for i in range(self.enemy_num)]
        self.action_type = options.action_type
        self.nn_controlled = False
        self.orders = Orders_set(self.num_robots)
        if self.action_type == 'Hybrid':
            self.actions = {'Continuous': {'x': [-2, 2], 'y': [-2, 2], 'rotate': [-2, 2]},
                            'Discrete': {'shoot': 2}}
        elif self.action_type == 'MultiDiscrete' or 'Discrete':
            self.actions = {'x': 3, 'y': 3, 'rotate': 3, 'shoot': 2}
            if self.enemy_num > 1:
                self.actions.update({'shoot_target': 2})
        self.individual_obs_range = {'x': [0, 808],
                                     'y': [0, 448],
                                     'hp': [0, 100],
                                     'angle': [-180, 180],
                                     # 'bullet': [0, 500],
                                     'vx': [-10, 10],
                                     'vy': [-10, 10]
                                     }

    def decode_actions(self, game_state, actions):
        # 检查actions是否在合理范围内
        if actions is not None:
            for i, action in enumerate(actions):
                if game_state.robots[self.robot_ids[i]].hp <= 0:
                    continue
                for j, action_key in enumerate(self.actions):
                    assert action[j] < self.actions[action_key], "action value out of bound"

    def get_robot_ids(self):
        return self.robot_ids

    def get_individual_obs_space(self, idx):
        obs_low = []
        obs_high = []
        assert idx in self.robot_ids, "current idx does not exit"

        for s in self.individual_obs_range:
            obs_low.append(self.individual_obs_range[s][0])
            obs_high.append(self.individual_obs_range[s][1])
        return obs_low, obs_high

    def get_individual_obs(self, idx, game_state):
        observations = []
        for i, s in enumerate(self.individual_obs_range):
            if game_state.robots_survival_status[idx] or idx in game_state.robots_killed_this_step:
                observations.append(eval('game_state.robots[' + str(idx) + '].' + s) / self.individual_obs_range[s][1])
            else:
                observations.append(0.0)
            game_state.robots[idx].robot_info_text[s] = observations[-1]
        return observations

    def reset(self):
        pass

class Orders(object):  # 指令
    def __init__(self, x=0, y=0, rotate=0,
                 shoot=0, yaw=0,
                 do_route_plan=False, dir_relate_to_map=False, swing=False, auto_rotate=False):
        self.x = x  # 启动路径规划时
        self.y = y
        self.rotate = rotate  # -1~1	底盘，-1：左转，0：不动，1：右转	a/d
        self.shoot = shoot  # 0~1	是否射击，0：否，1：是	space
        self.yaw = yaw  # -1~1	云台，-1：左转，0：不动，1：右转	b/m
        self.shoot_target_enemy = 0
        self.do_route_plan = do_route_plan
        self.stop = False
        self.freq_update_goal = 5
        self.dir_relate_to_map = dir_relate_to_map
        self.swing = swing

        self.auto_rotate = auto_rotate
        self.rotate_target_mode = False


class Orders_set(object):
    def __init__(self, num):
        self.set = []
        for n in range(num):
            self.set.append(Orders())
        self.sets_new = copy.deepcopy(self.set)

    def reset(self):
        self.set = copy.deepcopy(self.sets_new)

    def combine(self, orders0, orders1):
        self.set = orders0.set + orders1.set


def sign(action):
    if action > 0:
        return 1
    elif action == 0:
        return 0
    else:
        return -1

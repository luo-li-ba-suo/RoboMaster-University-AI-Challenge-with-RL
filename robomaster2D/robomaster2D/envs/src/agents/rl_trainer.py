from robomaster2D.envs.src.agents_common import *


class My_Agent(object):
    def __init__(self, _id, options):
        self.id = _id
        self.name = 'rl_trainer'
        self.robot_blue_num = options.robot_b_num
        self.robot_red_num = options.robot_r_num
        self.num_robots = options.robot_b_num if _id else options.robot_r_num
        self.enemy_num = options.robot_r_num if _id else options.robot_b_num
        # robot id 表示agent对应的robot在kernel_game中的索引
        self.robot_ids = [(i + self.robot_red_num if _id else i) for i in range(self.num_robots)]
        self.action_type = options.action_type
        self.orders = Orders_set(self.num_robots)
        # self.actions = {'x': 3, 'y': 3, 'rotate': 3, 'shoot_target_enemy': 2, 'shoot': 2}
        if self.action_type == 'Hybrid':
            self.actions = {'Continuous': {'x': [-2, 2], 'y': [-2, 2], 'rotate': [-2, 2]},
                            'Discrete': {'shoot': 2}}
        elif self.action_type == 'MultiDiscrete' or 'Discrete':
            self.actions = {'x': 3, 'y': 3, 'rotate': 3, 'shoot': 2}
            if self.enemy_num > 1:
                self.actions.update({'shoot_target': 2})

        # self.state = {'x': [0, 808],
        #               'y': [0, 448],
        #               'angle': [-180, 180],
        #               'yaw': [-90, 90],
        #               'hp': [0, 2000],
        #               'heat': [0, 400],
        #               'freeze_time[0]': [0, 2000],
        #               'freeze_time[1]': [0, 2000],
        #               'bullet': [0, 500]
        #               }
        self.state = {'x': [0, 808],
                      'y': [0, 448],
                      'hp': [0, 1000],
                      'angle': [-180, 180],
                      'vx': [-10, 10],
                      'vy': [-10, 10]
                      # 'bullet': [0, 500]
                      }

    def decode_actions(self, game_state, actions):  # 根据动作编码，解码产生动作
        self.orders.reset()
        action_offset = [1, 1, 1, 0] if self.enemy_num == 1 else [1, 1, 1, 0, 0]
        for i in range(self.num_robots):
            if game_state.robots[self.robot_ids[i]].hp <= 0:
                continue
            action = actions[i] - action_offset
            self.orders.set[i].x = action[0]
            self.orders.set[i].y = action[1]
            self.orders.set[i].rotate = action[2]
            if self.enemy_num > 1:
                self.orders.set[i].shoot_target_enemy = action[4]
            if game_state.robots[self.robot_ids[i]].aimed_enemy is not None:
                self.orders.set[i].shoot = action[3]
        return self.orders

"""
nn_agent


Author: DQ, HITSZ
Date: March 4th, 2022

nn_agent代表用神经网络控制的智能体，不训练
"""

from robomaster2D.envs.src.agents_common import *


class My_Agent(Base_Agent):
    def __init__(self, _id, options):
        super().__init__(_id, options)
        self.name = 'nn_enemy'

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

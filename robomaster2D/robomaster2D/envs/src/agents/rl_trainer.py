from robomaster2D.envs.src.agents_base import *


class My_Agent(Base_Agent):
    def __init__(self, _id, options):
        super().__init__(_id, options)
        self.name = 'rl_trainer'
        self.nn_controlled = True
        self.individual_obs_range.update({'enemy_hit_record.left.one_step': [0, 1],
                                          'enemy_hit_record.right.one_step': [0, 1],
                                          'enemy_hit_record.behind.one_step': [0, 1],
                                          'enemy_hit_record.front.one_step': [0, 1],
                                          'armor_hit_enemy_record.left.one_step': [0, 1],
                                          'armor_hit_enemy_record.right.one_step': [0, 1],
                                          'armor_hit_enemy_record.behind.one_step': [0, 1],
                                          'armor_hit_enemy_record.front.one_step': [0, 1],
                                          'wheel_hit_obstacle_record.one_step': [0, 1],
                                          'wheel_hit_wall_record.one_step': [0, 1],
                                          'wheel_hit_robot_record.one_step': [0, 1],
                                          'armor_hit_obstacle_record.left.one_step': [0, 1],
                                          'armor_hit_obstacle_record.right.one_step': [0, 1],
                                          'armor_hit_obstacle_record.behind.one_step': [0, 1],
                                          'armor_hit_obstacle_record.front.one_step': [0, 1],
                                          'armor_hit_robot_record.left.one_step': [0, 1],
                                          'armor_hit_robot_record.right.one_step': [0, 1],
                                          'armor_hit_robot_record.behind.one_step': [0, 1],
                                          'armor_hit_robot_record.front.one_step': [0, 1],
                                          })

    def decode_actions(self, game_state, actions):  # 根据动作编码，解码产生动作
        super().decode_actions(game_state, actions)
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

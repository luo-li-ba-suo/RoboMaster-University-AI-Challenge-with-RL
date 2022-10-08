from robomaster2D.envs.src.agents_base import *


class My_Agent(Base_Agent):
    def __init__(self, _id, options):
        super().__init__(_id, options)
        self.name = 'handcrafted_enemy'
        self.frame = 0

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
        return self.orders

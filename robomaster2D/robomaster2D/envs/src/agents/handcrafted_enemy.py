from robomaster2D.envs.src.agents_common import *


class My_Agent(Base_Agent):
    def __init__(self, _id, options):
        super().__init__(_id, options)
        self.name = 'handcrafted_enemy'
        self.frame = 0

    def Issue_Orders(self, game_state):  # 根据状态直接产生动作
        self.orders.reset()
        for event in game_state.KB_events:
            for i, key_board_order in enumerate(self.keyboard_orders):
                if key_board_order['forward'] in event:
                    self.orders.set[i].x += 1
                if key_board_order['backward'] in event:
                    self.orders.set[i].x -= 1
                if key_board_order['left'] in event:
                    self.orders.set[i].y -= 1
                if key_board_order['right'] in event:
                    self.orders.set[i].y += 1
                if key_board_order['left_rotate'] in event:
                    self.orders.set[i].rotate -= 1
                if key_board_order['right_rotate'] in event:
                    self.orders.set[i].rotate += 1
                if key_board_order['yaw_left'] in event:
                    self.orders.set[i].yaw -= 1
                if key_board_order['yaw_right'] in event:
                    self.orders.set[i].yaw += 1
                if key_board_order['shoot'] in event:
                    self.orders.set[i].shoot += 1
        return self.orders

    def decode_actions(self, game_state, actions=None):  # 根据动作编码，解码产生动作
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
        return self.orders

from robomaster2D.envs.src.agents_common import *


class My_Agent(Base_Agent):
    def __init__(self, _id, options):
        super().__init__(_id, options)
        self.name = 'handcrafted_enemy'
        self.frame = 0

    def decode_actions(self, game_state, actions=None):  # 根据动作编码，解码产生动作
        return self.orders

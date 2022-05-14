import copy
import random

class Orders(object):  # 指令
    def __init__(self, x=0, y=0, rotate=0,
                 shoot=0, yaw=0, auto_aim=True, auto_shoot=True,
                 do_route_plan=False, dir_relate_to_map=False, swing=False):
        self.x = x  # 启动路径规划时
        self.y = y
        self.rotate = rotate  # -1~1	底盘，-1：左转，0：不动，1：右转	a/d
        self.shoot = shoot  # 0~1	是否射击，0：否，1：是	space
        self.yaw = yaw  # -1~1	云台，-1：左转，0：不动，1：右转	b/m
        self.auto_aim = auto_aim  # 0~2	是否启用自瞄，0：否，1：自瞄模式，2：自动瞄准射击模式
        self.auto_shoot = auto_shoot  # 0~2	是否启用自瞄，0：否，1：自瞄模式，2：自动瞄准射击模式
        self.do_route_plan = do_route_plan
        self.freq_update_goal = 20
        self.dir_relate_to_map = dir_relate_to_map
        self.swing = swing
        self.shoot_target_enemy = -1

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



class My_Agent(object):
    def __init__(self, _id, options):
        self.id = _id
        self.name = 'handcrafted_enemy'
        self.robot_blue_num = options.robot_b_num
        self.robot_red_num = options.robot_r_num
        self.num_robots = options.robot_b_num if _id else options.robot_r_num
        self.num_enemy_robots = options.robot_r_num if _id else options.robot_b_num
        self.robot_ids = [(i + self.robot_red_num if _id else i) for i in range(self.num_robots)]
        self.orders = Orders_set(self.num_robots)
        self.keyboard_orders = [{'forward': 'K_w', 'backward': 'K_s', 'left': 'K_q', 'right': 'K_e',
                                 'left_rotate': 'K_a', 'right_rotate': 'K_d',
                                 'yaw_left': 'K_z', 'yaw_right': 'K_c', 'shoot': 'K_SPACE'},
                                {'forward': 'K_p', 'backward': 'K_SEMICOLON', 'left': 'K_o', 'right': 'K_LEFTBRACKET',
                                 'left_rotate': 'K_l', 'right_rotate': 'K_QUOTE',
                                 'yaw_left': 'K_PERIOD', 'yaw_right': 'K_SLASH', 'shoot': 'K_RSHIFT'},
                                {'forward': 'K_u', 'backward': 'K_j', 'left': 'K_y', 'right': 'K_i',
                                 'left_rotate': 'K_h', 'right_rotate': 'K_k',
                                 'yaw_left': 'K_n', 'yaw_right': 'K_COMMA', 'shoot': 'K_v'},
                                {'forward': 'K_KP8', 'backward': 'K_KP5', 'left': 'K_KP7', 'right': 'K_KP9',
                                 'left_rotate': 'K_KP4', 'right_rotate': 'K_KP6',
                                 'yaw_left': 'K_KP1', 'yaw_right': 'K_KP3', 'shoot': 'K_DOWN'}]
        self.keyboard_orders = self.keyboard_orders[2: self.num_robots + 2] if _id else \
            self.keyboard_orders[0:self.num_robots]  # 蓝色和红色各自专用按键
        self.action_type = options.action_type
        self.actions = None
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
                      # 'bullet': [0, 500]
                      }
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

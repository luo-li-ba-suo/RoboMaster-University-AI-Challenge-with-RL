import copy


class Orders(object):  # 指令
    def __init__(self, x=0, y=0, rotate=0,
                 shoot=0, yaw=0,
                 do_route_plan=False, dir_relate_to_map=False, swing=False):
        self.x = x  # 启动路径规划时
        self.y = y
        self.rotate = rotate  # -1~1	底盘，-1：左转，0：不动，1：右转	a/d
        self.shoot = shoot  # 0~1	是否射击，0：否，1：是	space
        self.yaw = yaw  # -1~1	云台，-1：左转，0：不动，1：右转	b/m
        self.shoot_target_enemy = 0
        self.do_route_plan = do_route_plan
        self.freq_update_goal = 20
        self.dir_relate_to_map = dir_relate_to_map
        self.swing = swing


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

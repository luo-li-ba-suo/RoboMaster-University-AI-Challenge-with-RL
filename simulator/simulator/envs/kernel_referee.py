import numpy as np


def get_enemy_index(state, n, m):
    if n < state.robot_r_num:
        m += state.robot_r_num
    return m


def get_armor_center(robot, i):
    rotate_matrix = np.array([[np.cos(-np.deg2rad(robot.angle + 90)), -np.sin(-np.deg2rad(robot.angle + 90))],
                              [np.sin(-np.deg2rad(robot.angle + 90)), np.cos(-np.deg2rad(robot.angle + 90))]])
    xs = np.array([[0, -30], [18.5, 0], [0, 30], [-18.5, 0]])
    return np.matmul(xs[i], rotate_matrix) + robot.center


def get_points_armor_vertex(robot):
    rotate_matrix = np.array([[np.cos(-np.deg2rad(robot.angle + 90)), -np.sin(-np.deg2rad(robot.angle + 90))],
                              [np.sin(-np.deg2rad(robot.angle + 90)), np.cos(-np.deg2rad(robot.angle + 90))]])
    return [np.matmul(x, rotate_matrix) + robot.center for x in robot.armors], \
           [np.matmul(x, rotate_matrix) + robot.center for x in robot.outlines]


def check_points_wheel(robot):
    rotate_matrix = np.array([[np.cos(-np.deg2rad(robot.angle + 90)), -np.sin(-np.deg2rad(robot.angle + 90))],
                              [np.sin(-np.deg2rad(robot.angle + 90)), np.cos(-np.deg2rad(robot.angle + 90))]])
    xs = np.array([[-22.5, -29], [22.5, -29],
                   [-22.5, -14], [22.5, -14],
                   [-22.5, 14], [22.5, 14],
                   [-22.5, 29], [22.5, 29]])
    return [np.matmul(xs[i], rotate_matrix) + robot.center for i in range(xs.shape[0])]


def get_robot_outline(robot):
    rotate_matrix = np.array([[np.cos(-np.deg2rad(robot.angle + 90)), -np.sin(-np.deg2rad(robot.angle + 90))],
                              [np.sin(-np.deg2rad(robot.angle + 90)), np.cos(-np.deg2rad(robot.angle + 90))]])
    xs = np.array([[-22.5, -30], [22.5, 30], [-22.5, 30], [22.5, -30]])
    return [np.matmul(xs[i], rotate_matrix) + robot.center for i in range(xs.shape[0])]


class Referee(object):
    def __init__(self, state, map, options):
        self.map = map
        self.state = state
        self.HP_reduction = {'behind': 60, 'front': 20, 'left': 40, 'right': 40, 'not_bullet': 10}
        self.collision_reduce_hp = options.collision_reduce_hp
        self.cooling_freq = options.cooling_freq
        self.overheating_protection = options.overheating_protection

    def check_armor(self, act_feedback):
        # act_feedback.robot_collision_to_source()
        for n in range(self.state.robot_num):
            for part in ['behind', 'front', 'left', 'right']:
                # 根据子弹来源记录和扣血
                BULLET_HIT = False
                OTHER_HIT = False
                for i in range(self.state.robot_num):
                    if n == i: continue
                    if 'BULLET' + str(i) in act_feedback.hit_source[n][part]:
                        BULLET_HIT = True
                        if self.state.robots[n].owner == self.state.robots[i].owner:
                            self.state.robots[n].armor_hit_teammate_record.add(part)
                            self.state.robots[i].teammate_hit_record.add(part)
                        else:
                            self.state.robots[n].armor_hit_enemy_record.add(part)
                            self.state.robots[i].enemy_hit_record.add(part)
                if 'ROBOT' in act_feedback.hit_source[n][part]:
                    OTHER_HIT = True
                    self.state.robots[n].armor_hit_robot_record.add(part)
                if 'OBSTACLE' in act_feedback.hit_source[n][part]:
                    OTHER_HIT = True
                    self.state.robots[n].armor_hit_obstacle_record.add(part)
                if 'BOUNDARY' in act_feedback.hit_source[n][part]:
                    OTHER_HIT = True
                    self.state.robots[n].armor_hit_wall_record.add(part)
                if BULLET_HIT:
                    self.state.robots[n].hp_loss.add(self.HP_reduction[part])
                elif OTHER_HIT and self.collision_reduce_hp:
                    self.state.robots[n].hp_loss.add(self.HP_reduction['not_bullet'])
            if 'ROBOT' in act_feedback.hit_source[n]['not_armor']:
                self.state.robots[n].wheel_hit_robot_record.add()
            if 'OBSTACLE' in act_feedback.hit_source[n]['not_armor']:
                self.state.robots[n].wheel_hit_obstacle_record.add()
            if 'BOUNDARY' in act_feedback.hit_source[n]['not_armor']:
                self.state.robots[n].wheel_hit_wall_record.add()
        act_feedback.reset()  # 在这里重置表示用过了才重置

    def buff_check(self):

        # 加成区判定
        for robot in self.state.robots:
            robot.reward_state = [0, 0]
        for n in range(self.state.robot_num):
            unused_buff_ids = np.where(self.state.buff[:, 1] == 1)[0]
            for unused_i in unused_buff_ids:
                a = self.map.buff_areas[unused_i]
                if a[0] <= self.state.robots[n].x <= a[1] and a[2] <= self.state.robots[n].y <= a[3]:
                    self.state.buff[unused_i, 1] = 0
                    # TODO:car n activated buff unused_i what will happen?
                    # hp_red
                    if self.state.buff[unused_i, 0] == 2:
                        for robot in self.state.robots:
                            if robot.owner == 0:  # red
                                if robot.hp == 0: continue
                                hp_add = 200 if robot.hp <= 1800 else 2000 - robot.hp
                                robot.hp_loss.add(-hp_add)
                                robot.reward_state[1] = 1
                                robot.buff_hp.add(hp_add)
                    # hp_blue
                    elif self.state.buff[unused_i, 0] == 0:
                        for robot in self.state.robots:
                            if robot.owner == 1:  # blue
                                if robot.hp == 0: continue
                                hp_add = 200 if robot.hp <= 1800 else 2000 - robot.hp
                                robot.hp_loss.add(-hp_add)
                                robot.reward_state[1] = 1
                                robot.buff_hp.add(hp_add)
                    ###bullet blue
                    elif self.state.buff[unused_i, 0] == 3:
                        for robot in self.state.robots:
                            if robot.owner == 0:
                                robot.bullet += 100
                                robot.reward_state[0] = 1
                                robot.buff_bullet.add(100)
                    ###bullet red
                    elif self.state.buff[unused_i, 0] == 1:
                        for robot in self.state.robots:
                            if robot.owner == 1:
                                robot.bullet += 100
                                robot.reward_state[0] = 1
                                robot.buff_bullet.add(100)
                    ###no shoot
                    elif self.state.buff[unused_i, 0] == 4:
                        self.state.robots[n].freeze_time[0] = 2000  # punish 10 seconds
                        self.state.robots[n].freeze_state[0] = 1
                    ###no move
                    else:
                        self.state.robots[n].freeze_time[1] = 2000  # punish 10 seconds
                        self.state.robots[n].freeze_state[1] = 1

    def cross(self, p1, p2, p3):
        # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
        x1 = p2[0] - p1[0]
        y1 = p2[1] - p1[1]
        x2 = p3[0] - p1[0]
        y2 = p3[1] - p1[1]
        return x1 * y2 - x2 * y1

    def segment(self, p1, p2, p3, p4):
        # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
        if (max(p1[0], p2[0]) >= min(p3[0], p4[0]) and max(p3[0], p4[0]) >= min(p1[0], p2[0])
                and max(p1[1], p2[1]) >= min(p3[1], p4[1]) and max(p3[1], p4[1]) >= min(p1[1], p2[1])):
            if (self.cross(p1, p2, p3) * self.cross(p1, p2, p4) <= 0 and self.cross(p3, p4, p1) * self.cross(p3, p4,
                                                                                                             p2) <= 0):
                return True
            else:
                return False
        else:
            return False

    def point_rect_check(self, p, r0, r1, r2, r3):
        if (self.cross(r0, r1, p) * self.cross(r3, r2, p) <= 0 and self.cross(r1, r2, p) * self.cross(r0, r3, p) <= 0):
            return True

    def line_rect_check(self, l1, l2, sq):
        # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
        # check if line cross rect, sq = [x_leftdown, y_leftdown, x_rightup, y_rightup]
        if (sq[0] <= l1[0] <= sq[2] and sq[1] <= l1[1] <= sq[3]) or \
                (sq[0] <= l2[0] <= sq[2] and sq[1] <= l2[1] <= sq[3]):
            return True
        p1 = [sq[0], sq[1]]  # 左下角
        p2 = [sq[2], sq[3]]  # 右上角
        p3 = [sq[2], sq[1]]  # 右下角
        p4 = [sq[0], sq[3]]  # 左上角
        if self.segment(l1, l2, p1, p2) or self.segment(l1, l2, p3, p4):
            return True
        else:
            return False

    def line_prismatic_check(self, l1, l2, sq):
        # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
        # check if line cross rect, sq = [x_leftdown, y_leftdown, x_rightup, y_rightup]
        # if self.cross()
        p1 = [sq[0], (sq[1] + sq[3]) / 2]
        p2 = [sq[2], (sq[1] + sq[3]) / 2]
        p3 = [(sq[0] + sq[2]) / 2, sq[1]]
        p4 = [(sq[0] + sq[2]) / 2, sq[3]]
        if self.point_rect_check(l1, p1, p3, p2, p4) or self.point_rect_check(l2, p1, p3, p2, p4):
            return True
        if self.segment(l1, l2, p1, p2) or self.segment(l1, l2, p3, p4):
            return True
        else:
            return False

    def line_barriers_check(self, l1, l2):
        for i, b in enumerate(self.map.barriers):

            sq = [b[0], b[2], b[1], b[3]]
            if i == 4:
                if self.line_prismatic_check(l1, l2, sq): return True
            else:
                if self.line_rect_check(l1, l2, sq): return True
        return False

    def line_robots_check(self, l1, l2):
        for robot in self.state.robots:
            if (robot.center == l1).all() or (robot.center == l2).all():
                continue
            p1, p2, p3, p4 = get_robot_outline(robot)
            if self.segment(l1, l2, p1, p2) or self.segment(l1, l2, p3, p4): return True
        return False

    def cooling_barrel(self):
        for n in range(self.state.robot_num):
            if self.overheating_protection:
                if self.state.robots[n].heat >= 200:
                    self.state.robots[n].cannot_shoot_overheating = True
                else:
                    self.state.robots[n].cannot_shoot_overheating = False
            if self.state.robots[n].heat >= 360:  # 超过360热量立即扣血
                hp_loss = (self.state.robots[n].heat - 360) * 40
                self.state.robots[n].hp_loss.add(hp_loss)
                self.state.robots[n].hp_loss_from_heat.add(hp_loss)
                self.state.robots[n].heat = 360
            if not self.state.frame % (200 // self.cooling_freq):

                if 360 > self.state.robots[n].heat > 240:  # 超过240热量扣血
                    hp_loss = (self.state.robots[n].heat - 240) * 4
                    self.state.robots[n].hp_loss.add(hp_loss)
                    self.state.robots[n].hp_loss_from_heat.add(hp_loss)
                self.state.robots[n].heat -= 120 // self.cooling_freq if self.state.robots[
                                                                             n].hp >= 400 else 240 // self.cooling_freq
                if self.state.robots[n].heat < 0: self.state.robots[n].heat = 0

    def transfer_to_robot_coordinate(self, points, n):
        pan_vecter = -np.array(self.state.robots[n].center)
        rotate_matrix = np.array(
            [[np.cos(np.deg2rad(self.state.robots[n].angle + 90)),
              -np.sin(np.deg2rad(self.state.robots[n].angle + 90))],
             [np.sin(np.deg2rad(self.state.robots[n].angle + 90)),
              np.cos(np.deg2rad(self.state.robots[n].angle + 90))]])
        return np.matmul(points + pan_vecter, rotate_matrix)

    def check_bullet(self, bullet, previous_center, act_feedback):
        # bullet wall check
        if bullet.center[0] <= 0 or bullet.center[0] >= self.map.map_length \
                or bullet.center[1] <= 0 or bullet.center[1] >= self.map.map_width: return True
        # bullet barrier check
        if self.line_barriers_check(bullet.center, previous_center): return True
        # bullet armor check
        for i in range(self.state.robot_num):
            if i == bullet.owner: continue
            if np.abs(np.array(bullet.center) - np.array(self.state.robots[i].center)).sum() < 52.5:
                points = self.transfer_to_robot_coordinate(np.array([bullet.center, previous_center]), i)
                if self.segment(points[0], points[1], [-18.5, -5], [-18.5, 6]):
                    act_feedback.hit_source[i]['left'].append('BULLET' + str(bullet.owner))
                    return True
                if self.segment(points[0], points[1], [18.5, -5], [18.5, 6]):
                    act_feedback.hit_source[i]['right'].append('BULLET' + str(bullet.owner))
                    return True
                if self.segment(points[0], points[1], [-5, 30], [5, 30]):
                    act_feedback.hit_source[i]['behind'].append('BULLET' + str(bullet.owner))
                    return True
                if self.segment(points[0], points[1], [-5, -30], [5, -30]):
                    act_feedback.hit_source[i]['front'].append('BULLET' + str(bullet.owner))
                    return True
                if self.line_rect_check(points[0], points[1], [-18, -29, 18, 29]):
                    act_feedback.hit_source[i]['not_armor'].append('BULLET' + str(bullet.owner))
                    return True
        return False

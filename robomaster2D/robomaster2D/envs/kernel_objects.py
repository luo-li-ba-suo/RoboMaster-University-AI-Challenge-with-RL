import numpy as np
import random


class Record(object):
    type = 'record'
    current = 0
    one_step = 0
    one_episode = 0

    def __init__(self, name='Nobody', only_add_positive=False):
        self.only_add_positive = only_add_positive
        self.name = name

    def add(self, num=1):
        self.current = num
        if self.only_add_positive and num < 0:
            return
        self.one_step += num
        self.one_episode += num

    def reset_current(self):
        self.current = 0

    def reset_one_step(self):
        self.current = 0
        self.one_step = 0

    def reset_one_episode(self):
        self.current = 0
        self.one_step = 0
        self.one_episode = 0

    def print_one_step(self):
        return self.one_episode


class Armor_Record(object):
    type = 'armor_record'

    def __init__(self, name='Nobody'):
        self.name = name
        self.left = Record('left')
        self.right = Record('right')
        self.front = Record('front')
        self.behind = Record('behind')

    def add(self, aromor_hit):
        if 'left' in aromor_hit:
            self.left.add()
        if 'right' in aromor_hit:
            self.right.add()
        if 'front' in aromor_hit:
            self.front.add()
        if 'behind' in aromor_hit:
            self.behind.add()

    def reset_current(self):
        for armmor in [self.left, self.right, self.front, self.behind]:
            armmor.reset_current()

    def reset_one_step(self):
        for armmor in [self.left, self.right, self.front, self.behind]:
            armmor.reset_one_step()

    def reset_one_episode(self):
        for armmor in [self.left, self.right, self.front, self.behind]:
            armmor.reset_one_episode()

    def print_one_step(self):
        parts = ''
        if self.right.one_step:
            parts += 'R,'
        if self.left.one_step:
            parts += 'L,'
        if self.front.one_step:
            parts += 'F,'
        if self.behind.one_step:
            parts += 'B,'
        return parts


class Multi_Record(object):
    type = 'multi_record'

    def __init__(self, record_group=['front', 'left', 'behind', 'right'], num_group=1, name=""):
        self.name = name
        self.records = [{} for n in range(num_group)]
        for record in self.records:
            for record_ in record_group:
                record[record_] = Record(record_)

    def add(self, record_name, add_num=1, id_group=0):
        self.records[id_group][record_name].add(add_num)

    def reset_current(self):
        for record in self.records:
            for name in record:
                record[name].reset_current()

    def reset_one_step(self):
        for record in self.records:
            for name in record:
                record[name].reset_one_step()

    def reset_one_episode(self):
        for record in self.records:
            for name in record:
                record[name].reset_one_episode()

    def print_one_step(self):
        string = ''
        for i, record in enumerate(self.records):
            for name in record:
                if record[name].one_step:
                    string += str(i) + '.' + name[0:2] + ','
        return parts


class Robot(object):
    armors = np.array([[-6.5, -28], [6.5, -28],  # behind
                       [-6.5, 28], [6.5, 28],  # front
                       [-18.5, -7], [-18.5, 6],  # left
                       [18.5, -7], [18.5, 6]  # right
                       ])
    outlines = np.array([[-22.5, -30], [22.5, -30],  # behind
                         [-22.5, 30], [22.5, 30],  # front
                         [-22.5, -30], [-22.5, 30],  # left
                         [22.5, -30], [22.5, 30]])  # right

    def __init__(self, robot_r_num, robot_num, owner=0, id=0, x=0, y=0, angle=0, bullet=50, vx=0, vy=0, yaw=0, hp=2000,
                 no_dying=False, frame_num_one_second=20.0):
        self.frame_num_one_second = frame_num_one_second
        robot_b_num = robot_num - robot_r_num
        self.owner = owner  # 队伍，0：红方，1：蓝方
        self.id = id
        if owner == 0:
            if robot_r_num == 1:
                friend = None
            elif robot_r_num == 2:
                if id == 0:
                    friend = 1
                else:
                    friend = 0
            self.enemy = [n for n in range(robot_r_num, robot_num)]
        else:
            if robot_b_num == 1:
                friend = None
            elif robot_b_num == 2:
                if id == robot_r_num:
                    friend = 1 + id
                else:
                    friend = id - 1
            self.enemy = [n for n in range(0, robot_r_num)]
        self.friend = friend
        self.x = x  # 以车的起始角落为原点，并使地图的全部落在正半轴
        self.vx = vx
        self.y = y
        self.vy = vy
        self.center = np.array([x, y])
        self.angle = angle  # 底盘绝对角度 -180~180  原点与上相同，极轴落在x轴正方向，向y轴正方向旋转的方向为正
        self.yaw = yaw  # 云台相对底盘角度 -90~90
        self.aimed_enemy = []  # 存放瞄准的敌人index
        self.heat = 0  # 枪口热度
        self.hp = hp  # 血量 0~2000
        self.no_dying = no_dying
        self.reward_state = [0, 0]
        # self.can_shoot = 1  # 决策频率高于出弹最高频率（10Hz）
        self.bullet = bullet  # 剩余子弹量
        self.bullet_speed = 25 * 100 / self.frame_num_one_second  # 子弹初速度 25m/s * 100cm/m / 20frame/s = 125cm/frame
        # 注意枪管发热热量增量应是子弹的原始速度
        # self.yaw_angle = 0

        # 运动物理参数
        self.speed_acceleration = 20 * 100 / self.frame_num_one_second / self.frame_num_one_second  # 加速度 20m/s^2 * 100cm/m / 20frame/s / 20frame/s = 5 cm/frame^2
        self.speed_max = 2 * 100 / self.frame_num_one_second  # 最大速度 2m/s * 100cm/m / 20frame/s = 10 cm/frame
        self.rotate_acceleration = 20 * 100 / self.frame_num_one_second / self.frame_num_one_second  # 5 deg/frame^2
        self.rotate_speed_max = 2 * 100 / self.frame_num_one_second  # 10 deg/frame
        self.drag_acceleration = 20 * 100 / self.frame_num_one_second / self.frame_num_one_second  # 5 cm/frame^2
        self.rotate_drag_acceleration = 20 * 100 / self.frame_num_one_second / self.frame_num_one_second  # 5 deg/frame^2

        # 由自瞄系统控制：
        self.yaw_acceleration = 20 * 100 / self.frame_num_one_second / self.frame_num_one_second  # 5 deg/frame^2
        self.yaw_rotate_speed_max = 4 * 100 / self.frame_num_one_second  # 20 deg/frame
        self.yaw_drag_acceleration = 20 * 100 / self.frame_num_one_second / self.frame_num_one_second  # 5 deg/frame^2
        # 已弃用：
        # self.motion = 6  # 移动的惯性感大小x
        # self.rotate_motion = 4  # 底盘旋转的惯性感大小
        # self.yaw_motion = 3  # 云台旋转的惯性感大小

        # self.camera_angle = 75 / 2  # 摄像头的视野范围
        self.camera_angle = 180 / 2  # 摄像头的视野范围
        self.move_discount = 2.6  # 撞墙之后反弹的强度大小
        self.lidar_angle = 120 / 2  # 激光雷达的视野视野范围

        # buff状态
        self.buff_hp = Record('buff补血')
        self.buff_bullet = Record('buff补弹')
        self.freeze_time = [0, 0]  # 惩罚完成剩余时间 需3s  0~600 以epoch为单位计算
        self.freeze_state = [0, 0]  # 0: 无冷却，1：无法射击，2：无法移动
        self.cannot_shoot_overheating = False
        # 以下状态为：每一次装甲板监测时更新；每一个step更新；每一个episode更新
        self.hp_loss_from_heat = Record('热量扣血')
        self.hp_loss = Record('扣血量', only_add_positive=True)
        self.bullet_out_record = Record('子弹消耗')
        self.wheel_hit_obstacle_record = Record('轮撞障碍')  # 轮子撞obstacle
        self.wheel_hit_wall_record = Record('轮撞墙')  # 轮子撞墙
        self.wheel_hit_robot_record = Record('轮撞他人')  # 轮子撞机器人
        self.armor_hit_robot_record = Armor_Record()  # 装甲板撞机器人
        self.armor_hit_wall_record = Armor_Record()  # 装甲板撞墙
        self.armor_hit_obstacle_record = Armor_Record()  # 装甲板撞obstacle
        self.armor_hit_enemy_record = Armor_Record()  # 被敌人击中
        self.armor_hit_teammate_record = Armor_Record()  # 被友军击中
        self.enemy_hit_record = Armor_Record()  # 击中敌人
        self.teammate_hit_record = Armor_Record()  # 击中友军
        self.total_record = [self.hp_loss,
                             self.hp_loss_from_heat,
                             self.bullet_out_record,
                             self.wheel_hit_wall_record,
                             self.wheel_hit_obstacle_record,
                             self.wheel_hit_robot_record,
                             self.armor_hit_robot_record,
                             self.armor_hit_wall_record,
                             self.armor_hit_obstacle_record,
                             self.armor_hit_enemy_record,
                             self.armor_hit_teammate_record,
                             self.enemy_hit_record,
                             self.teammate_hit_record]
        self.non_armor_record = [self.hp_loss,
                                 self.hp_loss_from_heat,
                                 self.bullet_out_record,
                                 self.wheel_hit_wall_record,
                                 self.wheel_hit_obstacle_record,
                                 self.wheel_hit_robot_record]
        # 装甲板信息
        self.armor = {'front': '', 'left': '', 'right': '', 'behind': ''}
        # 更多render内容：
        self.robot_info_text = {}
        # 更多plot内容：
        self.robot_info_plot = {}

    def reset_frame(self):
        self.aimed_enemy = []
        for record in self.total_record:
            record.reset_current()

    def reset_step(self):
        self.armor = {'front': '', 'left': '', 'right': '', 'behind': ''}
        for part in self.armor:
            if eval('self.armor_hit_robot_record.' + part + '.one_step'):
                self.armor[part] += '撞车'
            if eval('self.armor_hit_wall_record.' + part + '.one_step'):
                self.armor[part] += '撞墙'
            if eval('self.armor_hit_obstacle_record.' + part + '.one_step'):
                self.armor[part] += '撞碍'
            if eval('self.armor_hit_enemy_record.' + part + '.one_step'):
                self.armor[part] += '被击中'
        for record in self.total_record:
            record.reset_one_step()

    def reset_episode(self):
        for record in self.total_record:
            record.reset_one_episode()
        self.buff_hp.reset_one_episode()
        self.buff_bullet.reset_one_episode()

    def update_hp(self):
        self.hp -= self.hp_loss.current
        if self.hp <= 0:
            if self.no_dying[self.owner]:
                self.hp = 1
            else:
                self.hp = 0
        if self.hp > 2000:
            self.hp = 2000


class Bullet(object):
    def __init__(self, center, angle, speed=12.5, owner=0):
        self.bullet_speed = speed  # p/frame，子弹速度，单位为pixel
        self.center = center.copy()  # 机器人坐标
        self.center_original = center.copy()  # 机器人原始坐标
        # self.speed = speed
        self.angle = angle  # 机器人朝向与炮管朝向加和
        self.owner = owner
        self.journey = 0
        self.journey_max = 300
        self.disappear_check_interval = 1
        self.step = 0

    def disappear(self):
        if not self.step % self.disappear_check_interval:
            self.journey = ((self.center[0] - self.center_original[0]) ** 2 + (
                    self.center[1] - self.center_original[1]) ** 2) ** 0.5
        self.step += 1
        if self.journey > self.journey_max:
            return True if random.random() > 1/np.exp(800/self.journey_max) else False
        return False

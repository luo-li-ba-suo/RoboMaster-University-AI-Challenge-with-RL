import numpy as np
import matplotlib.pyplot as plt
import pygame
from robomaster2D.envs.src.buttons import Buttons
import os
import cv2
try:
    os.chdir('./robomaster2D/robomaster2D/envs')
except BaseException:
    pass


class Robot(pygame.sprite.Sprite):
    def __init__(self, idx, color, font):
        pygame.sprite.Sprite.__init__(self)
        self.idx = idx
        self.color = color
        self.font = font
        self.chassis_img = pygame.image.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                          './imgs/chassis_g.png'))
        self.gimbal_img = pygame.image.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                         './imgs/gimbal_g.png'))
        self.chassis_rotate = pygame.transform.rotate(self.chassis_img, 0)
        self.gimbal_rotate = pygame.transform.rotate(self.gimbal_img, 0)
        self.chassis_rect = self.chassis_rotate.get_rect()
        self.gimbal_rect = self.gimbal_rotate.get_rect()
        self.information = []
        self.information.append({'font': font.render('{}'.format(idx), False, color),  # 第二项为是否抗锯齿
                                 'coo_bias': [-20, -35], 'coo': None})
        self.information.append({'font': font.render('', False, color),  # 第二项为是否抗锯齿
                                 'coo_bias': [-20, 25], 'coo': None})

    def update(self, angle, yaw_angle, center, information):
        self.chassis_rotate = pygame.transform.rotate(self.chassis_img, -angle - 90)
        self.gimbal_rotate = pygame.transform.rotate(self.gimbal_img,
                                                     -yaw_angle - angle - 90)
        self.chassis_rect = self.chassis_rotate.get_rect()
        self.gimbal_rect = self.gimbal_rotate.get_rect()
        self.chassis_rect.center = center
        self.gimbal_rect.center = center
        if information is not None:
            for i in range(len(information)):
                self.information[i]['font'] = self.font.render(information[i], False, self.color)
                self.information[i]['coo'] = self.information[i]['coo_bias'] + center


class Buff(pygame.sprite.Sprite):
    def __init__(self, buff_img_path):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(buff_img_path)
        self.rect = self.image.get_rect()

    def update(self, center):
        self.rect.center = center


class User_Interface(object):
    def __init__(self, state, module_engine, orders, map, options):
        # 可切换参数：
        self.节能模式 = False
        self.训练模式 = options.training_mode
        self.高帧率模式 = True
        self.show_robot_points = False
        self.show_center_barrier_vertices = False
        self.show_goals_position = False
        self.show_goal_line = False
        self.show_state_data = True
        self.show_robot_data = True
        self.show_figure = True
        # self.
        # 将各个功能类的地址传进来
        self.state = state
        self.controller = module_engine
        self.show_poses = options.show_poses
        self.red_agent = options.red_agents_path
        self.blue_agent = options.blue_agents_path
        self.orders = orders
        self.map = map
        pygame.init()
        # 填色
        self.gray = (180, 180, 180)
        self.gray2 = (231, 230, 230)
        self.white = (255, 255, 255)
        self.rice = (163, 148, 128)
        self.red = (190, 20, 20)
        self.blue = (10, 125, 181)
        self.blue2 = (135, 206, 230)
        self.yellow = (255, 255, 0)
        self.green = (34, 139, 34)
        self.black = (0, 0, 0)
        self.background_color = self.gray2
        self.font_colors = [self.black, self.red]
        self.font = pygame.font.Font(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                  'SweiAxLegCJKtc-Black.ttf'), 13)

        self.screen_length = int(map.map_length * 2)
        self.screen_width = map.map_width * 2
        self.surface_background = pygame.Surface((self.screen_length, self.screen_width))
        self.surface_background_rect = self.surface_background.get_rect()
        # self.surface_background_rect.center = self.screen_width // 2, self.screen_length // 2
        self.robot_group = pygame.sprite.Group()
        self.buff_group = pygame.sprite.Group()
        self.render_background()
        self.initial_objects()
        button_figure = {'name': 'show_figure',  # 按键名字
                        'on_or_off': True,
                        'img_path': {'on_up': './imgs/buttons/on.png',  # 开状态
                                     'off_up': './imgs/buttons/off.png',  # 关状态
                                     'on_down': './imgs/buttons/on_down.png',  # 开状态-按下按键
                                     'off_down': './imgs/buttons/off_down.png'},  # 关状态-按下按键
                        'size_of_img': (50, 50),  # 最终显示图像的尺寸
                        'size_x_of_button': 50, 'size_y_of_button': 28,  # 最终显示按键的尺寸
                        'center': [1300, 650]}  # 按键中心位置
        button_pause = {'name': 'pause',  # 按键名字
                        'on_or_off': False,  # pause the game at the start
                        'img_path': {'on_up': './imgs/buttons/on.png',  # 开状态
                                     'off_up': './imgs/buttons/off.png',  # 关状态
                                     'on_down': './imgs/buttons/on_down.png',  # 开状态-按下按键
                                     'off_down': './imgs/buttons/off_down.png'},  # 关状态-按下按键
                        'size_of_img': (50, 50),  # 最终显示图像的尺寸
                        'size_x_of_button': 50, 'size_y_of_button': 28,  # 最终显示按键的尺寸
                        'center': [1300, 690]}  # 按键中心位置
        button_switch = {'name': 'show_robot_data',  # 按键名字
                         'on_or_off': True,
                         'img_path': {'on_up': './imgs/buttons/on.png',  # 开状态
                                      'off_up': './imgs/buttons/off.png',  # 关状态
                                      'on_down': './imgs/buttons/on_down.png',  # 开状态-按下按键
                                      'off_down': './imgs/buttons/off_down.png'},  # 关状态-按下按键
                         'size_of_img': (50, 50),  # 最终显示图像的尺寸
                         'size_x_of_button': 50, 'size_y_of_button': 28,  # 最终显示按键的尺寸
                         'center': [1300, 730]}  # 按键中心位置
        button_reset = {'name': 'reset',  # 按键名字
                        'img_path': {'on_up': './imgs/buttons/reset.png',  # 开状态
                                     'off_up': './imgs/buttons/reset.png',  # 关状态
                                     'on_down': './imgs/buttons/reset_down.png',  # 开状态-按下按键
                                     'off_down': './imgs/buttons/reset_down.png'},  # 关状态-按下按键
                        'size_of_img': (40, 36),  # 最终显示图像的尺寸
                        'size_x_of_button': 50, 'size_y_of_button': 28,  # 最终显示按键的尺寸
                        'center': [1300, 50]}  # 按键中心位置
        button_energy_saving = {'name': 'energy_saving',  # 按键名字
                                'on_or_off': False,  # energy saving at the start
                                'img_path': {'on_up': './imgs/buttons/on.png',  # 开状态
                                             'off_up': './imgs/buttons/off.png',  # 关状态
                                             'on_down': './imgs/buttons/on_down.png',  # 开状态-按下按键
                                             'off_down': './imgs/buttons/off_down.png'},  # 关状态-按下按键
                                'size_of_img': (50, 50),  # 最终显示图像的尺寸
                                'size_x_of_button': 50, 'size_y_of_button': 28,  # 最终显示按键的尺寸
                                'center': [1300, 770]}  # 按键中心位置
        button_increase_frame_rate = {'name': 'increase_frame_rate',  # 按键名字
                                      'on_or_off': self.高帧率模式,
                                      'img_path': {'on_up': './imgs/buttons/on.png',  # 开状态
                                                   'off_up': './imgs/buttons/off.png',  # 关状态
                                                   'on_down': './imgs/buttons/on_down.png',  # 开状态-按下按键
                                                   'off_down': './imgs/buttons/off_down.png'},  # 关状态-按下按键
                                      'size_of_img': (50, 50),  # 最终显示图像的尺寸
                                      'size_x_of_button': 50, 'size_y_of_button': 28,  # 最终显示按键的尺寸
                                      'center': [1300, 810]}  # 按键中心位置
        button_training = {'name': 'training',  # 按键名字
                           'on_or_off': self.训练模式,
                           'img_path': {'on_up': './imgs/buttons/on.png',  # 开状态
                                        'off_up': './imgs/buttons/off.png',  # 关状态
                                        'on_down': './imgs/buttons/on_down.png',  # 开状态-按下按键
                                        'off_down': './imgs/buttons/off_down.png'},  # 关状态-按下按键
                           'size_of_img': (50, 50),  # 最终显示图像的尺寸
                           'size_x_of_button': 50, 'size_y_of_button': 28,  # 最终显示按键的尺寸
                           'center': [1300, 850]}  # 按键中心位置
        buttons = [button_figure, button_pause, button_switch,
                   button_reset, button_energy_saving,
                   button_increase_frame_rate, button_training]
        for button in buttons:
            for key in button['img_path']:
                button['img_path'][key] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       button['img_path'][key])
        self.buttons = Buttons(buttons)
        self.button_id = {button['name']: i for i, button in enumerate(buttons)}
        self.feedback_UI = {'reset': [], 'reset_frame': [], 'continue': False}
        self.screen = pygame.display.set_mode((self.screen_length, self.screen_width))
        current_path = os.path.dirname(os.path.realpath(__file__))
        pygame.display.set_caption('ICRA - Robomaster University AI Challenge 2021 - Simulator - Version:'
                                   + current_path.split('/')[-7])
        self.text_training_state = "正在采样..."
        pygame.font.init()
        self.plot_size = 2
        self.clock = pygame.time.Clock()

        # 交互事件触发类型
        self.events_type = ['K_0', 'K_1', 'K_2', 'K_3', 'K_4', 'K_5', 'K_6', 'K_7', 'K_8', 'K_9',
                            'K_KP0', 'K_KP1', 'K_KP2', 'K_KP3', 'K_KP4', 'K_KP5', 'K_KP6', 'K_KP7', 'K_KP8', 'K_KP9',
                            'K_a', 'K_b', 'K_c',
                            'K_d', 'K_e', 'K_f', 'K_g', 'K_h', 'K_i', 'K_j', 'K_k', 'K_l', 'K_m', 'K_n', 'K_o', 'K_p',
                            'K_q',
                            'K_r', 'K_s', 'K_t', 'K_u', 'K_v', 'K_w', 'K_x', 'K_y', 'K_z',
                            'K_SPACE', 'K_LCTRL', 'K_RCTRL', 'K_LALT', 'K_RALT', 'K_COMMA', 'K_QUOTE', 'K_RSHIFT',
                            'K_LSHIFT',
                            'K_UP', 'K_DOWN', 'K_LEFT', 'K_RIGHT', 'K_LEFTBRACKET',
                            'K_PERIOD', 'K_SEMICOLON', 'K_SLASH',
                            'MOUSEBUTTONUP']
        self.keyboard_events_state = {}
        for event in self.events_type:
            self.keyboard_events_state[event] = False
        self.Mouse_events = {'mouse_down': [], 'mouse_up': []}
        self.single_input = options.single_input

        # 图表
        self.update_figure_interval = 5
        self.figures_to_show = []
        self.do_plot = options.do_plot


    def update_display(self):
        if self.高帧率模式:
            self.state.render_per_frame = 1
        else:
            self.state.render_per_frame = 600
        if self.训练模式:
            self.state.render_per_frame = 600
            self.screen.fill(self.gray)
            self.buttons.render(self.screen, 'training')
            self.update_state_data()
        elif self.节能模式:
            self.screen.fill(self.gray)
            self.update_buff()
            self.update_objects()
            self.buttons.render(self.screen, 'energy_saving')
            self.update_state_data()
        else:
            self.screen.blit(self.surface_background, self.surface_background_rect)  # 加载背景图片
            self.update_buff()
            self.update_objects()
            self.update_state_data()
            info = self.font.render('机器人信息', False, self.font_colors[0])
            self.screen.blit(info, (self.map.map_length + 30, 4 + 5 * 17))

            if self.show_robot_data:
                self.update_robot_data()
            if self.show_figure:
                self.update_figure()
            if self.show_goal_line:
                self.update_goal_line()
            self.buttons.render(self.screen)
            self.buttons.show_button_name(self.font, self.screen)
        pygame.display.flip()

    def render_background(self):
        # 加载barriers图片
        self.barriers2pics = {0: 'horizontal_1', 1: 'horizontal_2', 2: 'vertical_1', 3: 'horizontal_1', 4: 'center',
                              5: 'horizontal_1', 6: 'vertical_1', 7: 'horizontal_2', 8: 'horizontal_1'}
        self.barriers_img = []
        self.barriers_rect = []
        self.surface_background.fill(self.background_color)
        pygame.draw.rect(self.surface_background, self.gray,
                         [0, 0, self.map.map_length, self.map.map_width], 0)
        for i in range(self.map.barriers.shape[0]):
            self.barriers_img.append(
                pygame.image.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               './imgs/barrier_{}.png'.format(self.barriers2pics[i]))))
            self.barriers_rect.append(self.barriers_img[-1].get_rect())
            self.barriers_rect[-1].center = [self.map.barriers[i][0:2].mean(), self.map.barriers[i][2:4].mean()]
        # 旋转中间的障碍块
        if self.map.barriers.any():
            self.barriers_img[4] = pygame.transform.rotate(self.barriers_img[4], 45)
            self.barriers_rect[4] = self.barriers_img[4].get_rect()
            self.barriers_rect[4].center = [self.map.barriers[4][0:2].mean(), self.map.barriers[4][2:4].mean()]
        # load start areas imgs
        self.start_areas_img = []
        self.start_areas_rect = []
        for oi, o in enumerate(['red', 'blue']):
            for ti, t in enumerate(['start', 'start']):
                self.start_areas_img.append(pygame.image.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                           'imgs/area_{}_{}.png'.format(t, o))))
                self.start_areas_rect.append(self.start_areas_img[-1].get_rect())
                self.start_areas_rect[-1].center = [self.map.start_areas[oi, ti][0:2].mean(),
                                                    self.map.start_areas[oi, ti][2:4].mean()]
        for i in range(len(self.barriers_rect)):
            self.surface_background.blit(self.barriers_img[i], self.barriers_rect[i])
        for i in range(len(self.start_areas_rect)):
            self.surface_background.blit(self.start_areas_img[i], self.start_areas_rect[i])
        # 绘制背景图片
        self.background_img_path = [os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 'imgs/background/crosswise.png'),
                                    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 'imgs/background/vertical.png')]
        self.background_img_coor = [[[405, 672]], [[1011, 494], [1410, 494]]]
        self.background_img_resize = [[[808, 448]], [[400, 800], [400, 800]]]
        for i, img_coors in enumerate(self.background_img_coor):
            for j, img_coor in enumerate(img_coors):
                background_img = pygame.transform.scale(pygame.image.load(self.background_img_path[i]),
                                                        self.background_img_resize[i][j])
                background_img_rect = background_img.get_rect()
                background_img_rect.center = img_coor
                self.surface_background.blit(background_img, background_img_rect)
        if self.show_center_barrier_vertices:
            pygame.draw.circle(self.surface_background, [166, 0, 166], [self.map.barriers[4][0], 224], 1, 1)
            pygame.draw.circle(self.surface_background, [166, 0, 166], [self.map.barriers[4][1], 224], 1, 1)
            pygame.draw.circle(self.surface_background, [255, 255, 255], [404, self.map.barriers[4][2]], 1, 1)
            pygame.draw.circle(self.surface_background, [166, 0, 166], [404, self.map.barriers[4][3]], 1, 1)

    def initial_objects(self):
        self.bullet_img = pygame.image.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                         './imgs/bullet_s.png'))
        self.bullet_rect = self.bullet_img.get_rect()
        for n in range(self.state.robot_num):
            self.robot_group.add(Robot(n,
                                       self.blue if self.state.robots[n].owner else self.red,
                                       self.font))
        if self.state.buff is not None:
            # load buff areas imgs
            for oi, o in enumerate(
                    ['hp_blue', 'bullet_blue', 'hp_red', 'bullet_red', 'no_shoot', 'no_move', 'no_buff']):
                self.buff_group.add(Buff(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                      './imgs/area_{}.png'.format(o))))

    def update_buff(self):
        if self.state.buff is not None:
            for i in range(len(self.state.buff)):
                if self.state.buff[i, 1] == 0:  # the buff has been used
                    if self.节能模式:
                        continue
                    buff = self.buff_group.sprites()[-1]
                else:
                    buff = self.buff_group.sprites()[self.state.buff[i, 0]]
                buff.update([self.map.buff_areas[i][0:2].mean(), self.map.buff_areas[i][2:4].mean()])
                self.screen.blit(buff.image, buff.rect)

    def update_objects(self):
        for i in range(len(self.state.bullets)):
            self.bullet_rect.center = self.state.bullets[i].center
            self.screen.blit(self.bullet_img, self.bullet_rect)

        for n in range(self.state.robot_num):
            self.robot_group.sprites()[n].update(self.state.robots[n].angle,
                                                 self.state.robots[n].yaw,
                                                 self.state.robots[n].center,
                                                 ['{} | {}'.format(int(self.state.robots[n].hp), n + 1),
                                                  '{} {}'.format(int(self.state.robots[n].bullet),
                                                                 int(self.state.robots[n].heat))])
            self.screen.blit(self.robot_group.sprites()[n].chassis_rotate, self.robot_group.sprites()[n].chassis_rect)
            self.screen.blit(self.robot_group.sprites()[n].gimbal_rotate, self.robot_group.sprites()[n].gimbal_rect)
            if self.show_robot_points:
                [pygame.draw.circle(self.screen, [166, 0, 0], point, 1, 1) for point in
                 self.controller.points_for_render[n]]
            for info in self.robot_group.sprites()[n].information:
                self.screen.blit(info['font'], info['coo'])

    def update_state_data(self):
        if self.训练模式:
            info = self.font.render('训练模式 时间: {} 渲染间隔：{}'.format(self.state.time, self.state.render_per_frame), False,
                                    self.font_colors[0])
            self.screen.blit(info, (self.map.map_length + 30, 27))
        elif self.节能模式:
            info = self.font.render('节能模式 时间: {}'.format(self.state.time), False, self.font_colors[0])
            self.screen.blit(info, (self.map.map_length + 30, 27))
        else:
            info = self.font.render('当前状态：{}'.format(self.text_training_state), False, self.font_colors[0])
            self.screen.blit(info, (self.map.map_length + 30, 5))
            tags_state = {'时间': self.state.time,
                          '渲染间隔': self.state.render_per_frame,
                          '红方': self.red_agent,
                          '蓝方': self.blue_agent}
            for i, tag in enumerate(tags_state):
                info = self.font.render('{}: {}'.format(tag, tags_state[tag]), False, self.font_colors[0])
                self.screen.blit(info, (self.map.map_length + 30, 22 + i * 17))

            tags_state = {'frame': self.state.frame}
            for i, tag in enumerate(tags_state):
                info = self.font.render('{}: {}'.format(tag, tags_state[tag]), False, self.font_colors[0])
                self.screen.blit(info, (self.map.map_length + 130, 22 + i * 17))
            tags_state = {'红方胜率': round(self.state.r_win_record.get_win_rate(), 3),
                          '平局率': round(self.state.r_win_record.get_draw_rate(), 3)}
            for i, tag in enumerate(tags_state):
                info = self.font.render('{}: {}'.format(tag, tags_state[tag]), False, self.font_colors[0])
                self.screen.blit(info, (self.map.map_length + 230, 22 + i * 17))

    def update_robot_data(self):
        info = self.font.render(
            '        ' + ''.join([' robot ' + str(n) + '        ' for n in range(self.state.robot_num)]),
            False, self.font_colors[0])
        self.screen.blit(info, (self.map.map_length + 60, 10 + 6 * 17))

        for n, robot in enumerate(self.state.robots):
            tags_robots = {'idx': n, 'team': robot.owner,
                           'x': round(robot.x, 2), 'y': round(robot.y, 2),
                           'angle': round(robot.angle, 1), '云台角度': round(robot.yaw, 1),
                           'heat': robot.heat,
                           '冷却时间': robot.freeze_time,
                           '冷却状态': robot.freeze_state,
                           '子弹': robot.bullet,
                           # 'wheel_hit': robot.wheel_hit,
                           # 'armor_hit': robot.armor_hit,
                           # 'robot_hit': robot.robot_hit,
                           'x速度': '%.2f' % robot.vx,
                           'y速度': '%.2f' % robot.vy,
                           '左装甲板': robot.armor['left'],
                           '右装甲板': robot.armor['right'],
                           '前装甲板': robot.armor['front'],
                           '后装甲板': robot.armor['behind'],
                           '击敌左': robot.enemy_hit_record.left.one_episode,
                           '击敌右': robot.enemy_hit_record.right.one_episode,
                           '击敌前': robot.enemy_hit_record.front.one_episode,
                           '击敌后': robot.enemy_hit_record.behind.one_episode,
                           'buff_hp': robot.buff_hp.one_episode,
                           'buff_bullet': robot.buff_bullet.one_episode,
                           '动作': self.controller.orders_text[n]}

            for record in robot.non_armor_record:
                tags_robots[record.name] = record.print_one_step()
            for robot_info in robot.robot_info_text:
                tags_robots[robot_info] = '%.2f' % (robot.robot_info_text[robot_info])

            print_column = self.map.map_length
            print_row = 129
            for i, tag in enumerate(tags_robots):
                if i > 42:
                    print_column = self.map.map_length + 404
                    print_row = -605
                if ~n:  # 首先打印标签
                    info = self.font.render('{}'.format(tag), False, self.font_colors[1])
                    self.screen.blit(info, (print_column + 20, print_row + i * 17))
                # 打印各机器人信息
                info = self.font.render('{}'.format(tags_robots[tag]), False, (0, 0, 0))
                self.screen.blit(info, (print_column + 98 + n * 80, print_row + i * 17))

    def update_figure(self):
        if not self.state.frame % self.update_figure_interval:
            self.figures_to_show = []
            # 画柱状图
            hps = []
            label_hp = {'x': [], 'y': [], 'color': []}
            for n in range(self.state.robot_num):
                hps.append(self.state.robots[n].hp)
                label_hp['x'].append('blue' + str(n) if self.state.robots[n].owner else 'red' + str(n))
                label_hp['color'].append('b' if self.state.robots[n].owner else 'r')
                label_hp['y'] = 'hp'
            figure = self.bar([hps], [label_hp])
            plt.yticks([0, 400, 800, 1200, 1600, 2000])
            figure.canvas.draw()
            figure = np.array(figure.canvas.renderer._renderer)[:, :, 0:3]
            plt.close()
            self.figures_to_show.append(pygame.pixelcopy.make_surface(figure.transpose(1, 0, 2)))
            # 画折线图
            for i in range(self.state.robot_num):
                if not self.do_plot[i]: continue
                for plot_name in self.state.robots[i].robot_info_plot:
                    figure = plt.figure(figsize=(self.plot_size * 1.2, self.plot_size))
                    plt.plot(self.state.robots[i].robot_info_plot[plot_name])
                    plt.title(plot_name)
                    figure.canvas.draw()
                    figure = np.array(figure.canvas.renderer._renderer)[:, :, 0:3]
                    plt.close()
                    self.figures_to_show.append(pygame.pixelcopy.make_surface(figure.transpose(1, 0, 2)))
                # 画局部地图
                local_map = self.state.robots[i].local_map
                local_map_ = np.zeros_like(local_map[0, :, :], dtype=np.uint8)
                local_map_[np.where(local_map[0] == 0)] = 255
                local_map_ = local_map_.T
                local_map_ = cv2.resize(local_map_, (220,210))
                self.figures_to_show.append(pygame.pixelcopy.make_surface(local_map_))


        for i, figure in enumerate(self.figures_to_show):
            if i < 3:
                self.screen.blit(figure, (20 + 268*i, self.map.map_width + 20))
            else:
                self.screen.blit(figure, (20 + 268 * (i-3), self.map.map_width + 230))

    def bar(self, ys, labels):  # 传入n组数据，xs，ys分别为横纵数据集，labels为x,y标签集
        n = len(ys)
        fig = plt.figure(figsize=(self.plot_size * (n * 1.2), self.plot_size))
        plt.subplots_adjust(left=0.25)
        for i in range(n):
            # plt.subplot(1, n, i + 1)
            plt.title(labels[i]['y'])
            plt.bar(range(len(ys[i])), ys[i], color=labels[i]['color'], tick_label=labels[i]['x'])

        return fig

    def update_goal_line(self):
        if self.controller.route_plan is not None:
            for n, goal in enumerate(self.controller.route_plan.goals):
                pygame.draw.aaline(self.screen, self.blue if self.state.robots[n].owner else self.red,
                                   self.state.robots[0].center,
                                   self.controller.route_plan.goals[n], 1)

    def update_other_image(self, coor, image_path='./imgs/area_destination.png'):
        if coor is not None and image_path is not None:  # 如果赋值了目标点
            dest_image = pygame.image.load(image_path)
            dest_rect = dest_image.get_rect()
            dest_rect.center = coor
            self.screen.blit(dest_image, dest_rect)

    def update_events(self):
        self.state.KB_events = []
        self.Mouse_events = {'mouse_down': [], 'mouse_up': []}
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.state.do_render = False
                return True
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.Mouse_events['mouse_down'].append(event.pos)
            if event.type == pygame.MOUSEBUTTONUP:
                self.Mouse_events['mouse_up'].append(event.pos)
        # 按钮的响应内容
        self.buttons.update(self.Mouse_events)
        self.show_robot_data = self.buttons.buttons_[self.button_id['show_robot_data']]['on_or_off']
        self.show_figure = self.buttons.buttons_[self.button_id['show_figure']]['on_or_off']
        self.节能模式 = self.buttons.buttons_[self.button_id['energy_saving']]['on_or_off']
        self.训练模式 = self.buttons.buttons_[self.button_id['training']]['on_or_off']
        self.高帧率模式 = self.buttons.buttons_[self.button_id['increase_frame_rate']]['on_or_off']
        self.state.pause = self.buttons.buttons_[self.button_id['pause']]['on_or_off']
        if self.buttons.button_down_finish[self.button_id['pause']] != [] and self.state.pause is False:
            self.buttons.button_down_finish[self.button_id['pause']] = []
            self.feedback_UI['reset_frame'].append(True)
        if self.buttons.button_down_finish[self.button_id['reset']] != []:
            self.buttons.button_down_finish[self.button_id['reset']] = []
            self.feedback_UI['reset'].append(True)
        # 键盘的响应内容
        pressed = pygame.key.get_pressed()
        if self.single_input:
            new_pressed = False
        for event_type in self.events_type:
            if self.single_input:
                if not self.keyboard_events_state[event_type] and pressed[eval('pygame.' + event_type)]:
                    self.feedback_UI['continue'] = True
                    self.state.KB_events.append(event_type)
                    self.keyboard_events_state[event_type] = True
                    new_pressed = True
                elif not pressed[eval('pygame.' + event_type)]:
                    self.keyboard_events_state[event_type] = False

            else:
                if pressed[eval('pygame.' + event_type)]:
                    self.state.KB_events.append(event_type)
        if self.single_input:
            if not new_pressed:
                self.feedback_UI['continue'] = False

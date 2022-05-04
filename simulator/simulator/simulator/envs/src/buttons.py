"""
buttons module for pygame


Author: DQ, HITSZ
Date: July 23th, 2021

一个类统一管理多个按键
支持切换按钮、切换初始值

"""
import pygame
import sys


class Buttons(object):
    def __init__(self, buttons):
        self.buttons_ = []
        self.show_button_rect = False
        self.have_button_down = []
        self.have_button_up = []
        self.buttons_num = len(buttons)
        self.button_down_finish = [[] for _ in range(self.buttons_num)]
        for button in buttons:
            if 'on_or_off' in button:
                on_or_off = button['on_or_off']
            else:
                on_or_off = False
            images = {'on_up': pygame.image.load(button['img_path']['on_up']),
                      'on_down': pygame.image.load(button['img_path']['on_down'])}
            if 'off_up' in button['img_path']:
                images['off_up'] = pygame.image.load(button['img_path']['off_up'])
                images['off_down'] = pygame.image.load(button['img_path']['off_down'])
            else:
                images['off_up'] = images['on_up']
                images['off_down'] = images['on_down']
            for img in images:
                images[img] = pygame.transform.scale(images[img], button['size_of_img'])
            img_rect = images['on_up'].get_rect()
            button_rect = pygame.Rect(0, 0, button['size_x_of_button'], button['size_y_of_button'])
            img_rect.center = button['center']
            button_rect.center = button['center']
            self.buttons_.append({'images': images, 'img_rect': img_rect, 'button_rect': button_rect,
                                  'name': button['name'],
                                  'on_or_off': on_or_off, 'button_down': False, 'button_down_last_time': False})

    def update(self, events):
        self.have_button_down = []
        self.have_button_up = []
        for mouse_pos in events['mouse_down']:
            for i, button in enumerate(self.buttons_):
                if button['button_rect'].collidepoint(mouse_pos):
                    self.have_button_down.append(i)
        for mouse_pos in events['mouse_up']:
            for i, button in enumerate(self.buttons_):
                if button['button_rect'].collidepoint(mouse_pos):
                    self.have_button_up.append(i)
        for n in range(self.buttons_num):
            if n in self.have_button_down:
                self.buttons_[n]['button_down'] = True
            if n in self.have_button_up:
                self.buttons_[n]['button_down'] = False
            if self.buttons_[n]['button_down_last_time'] and not self.buttons_[n]['button_down']:
                self.buttons_[n]['on_or_off'] = not self.buttons_[n]['on_or_off']
                self.button_down_finish[n].append(True)
                if len(self.button_down_finish[n]) > 10:
                    self.button_down_finish[n] = [True]
            self.buttons_[n]['button_down_last_time'] = self.buttons_[n]['button_down']
        self.have_button_down = []
        self.have_button_up = []

    def render(self, screen, only_button = None):
        for n in range(self.buttons_num):
            if self.buttons_[n]['name'] != only_button and only_button != None:
                continue
            if self.buttons_[n]['on_or_off']:
                if self.buttons_[n]['button_down']:
                    screen.blit(self.buttons_[n]['images']['on_down'], self.buttons_[n]['img_rect'])
                else:
                    screen.blit(self.buttons_[n]['images']['on_up'], self.buttons_[n]['img_rect'])
            else:
                if self.buttons_[n]['button_down']:
                    screen.blit(self.buttons_[n]['images']['off_down'], self.buttons_[n]['img_rect'])
                else:
                    screen.blit(self.buttons_[n]['images']['off_up'], self.buttons_[n]['img_rect'])
            if self.show_button_rect:
                pygame.draw.rect(screen, [255, 0, 0], self.buttons_[n]['button_rect'], 5)

    def show_button_name(self, font, screen):
        for button in self.buttons_:
            info = font.render(button['name'], False, [0,0,0])
            screen.blit(info, (button['button_rect'].x + 50, button['button_rect'].y))

def main():
    pygame.init()
    clock = pygame.time.Clock()
    fps = 60
    size = [500, 500]
    bg = [255, 255, 255]

    button = {'name': 'switch',  # 按键名字
              'img_path': {'on_up': 'on.png',  # 开状态
                           'off_up': 'off.png',  # 关状态
                           'on_down': 'on_down.png',  # 开状态-按下按键
                           'off_down': 'off_down.png'},  # 关状态-按下按键
              'size_of_img': (100, 100),  # 最终显示图像的尺寸
              'size_x_of_button': 100, 'size_y_of_button': 50,  # 最终显示按键的尺寸
              'center': [200, 200]}  # 按键中心位置
    buttons = Buttons([button])

    screen = pygame.display.set_mode(size)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            buttons.check(event)
        screen.fill(bg)
        buttons.update(screen)
        pygame.display.update()
        clock.tick(fps)

    pygame.quit()
    sys.exit


if __name__ == '__main__':
    main()

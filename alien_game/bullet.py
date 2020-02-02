import pygame

class Bullet():
    def __init__(self,screen,ship):

        #初始化子弹的大小和颜色
        self.bullet_speed_factor=1
        self.bullet_width=3
        self.bullet_height=15
        self.bullet_color=[255,0,0]
        #创建表示子弹的矩形，再设置正确的位置
        self.screen=screen
        self.bullet_rect=pygame.Rect(0,0,self.bullet_width,self.bullet_height)
        self.bullet_rect.centerx=ship.image_rect.centerx
        self.bullet_rect.bottom=ship.image_rect.top
        #子弹的位置
        self.position=float(self.bullet_rect.top)


    def update(self):
        self.position-=self.bullet_speed_factor
        self.bullet_rect.top=self.position

    def plot_bullet(self):
        pygame.draw.rect(self.screen,self.bullet_color
                         ,self.bullet_rect)




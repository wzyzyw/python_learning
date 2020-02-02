import pygame

class Ship():
    #初始化飞船并设置其初始位置
    def __init__(self,screen):
        self.screen= screen
        #读取飞船图片
        self.image=pygame.image.load('image/rocket.bmp')
        #将飞船初始位置设置为屏幕底部中央
        self.image_rect=self.image.get_rect()
        self.screen_rect=self.screen.get_rect()
        self.image_rect.centerx = self.screen_rect.centerx
        self.image_rect.bottom = self.screen_rect.bottom

        #飞船移动标志位
        self.moving_right=False
        self.moving_left=False

        #飞船移动速度
        self.ship_speed_factor = 1.5


    def plot_ship(self):
        self.screen.blit(self.image,self.image_rect)

    def update(self):
        if self.moving_right and self.image_rect.right<self.screen_rect.right:
            self.image_rect.centerx+=self.ship_speed_factor
        elif self.moving_left and self.image_rect.left>self.screen_rect.left:
            self.image_rect.centerx-=self.ship_speed_factor
    #将飞船移动到中央
    def ship_center(self):
        self.image_rect.centerx=self.screen_rect.centerx

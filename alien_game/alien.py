import pygame
class Alien():
    def __init__(self,screen):
        self.screen=screen
        #加载外星人图片，并获取rect属性
        self.image=pygame.image.load('image/alien.bmp')
        self.alien_rect=self.image.get_rect()
        #每个外星人都在左上角
        self.alien_rect.x=self.alien_rect.width
        self.alien_rect.top=self.alien_rect.height
        #外星人速度和方向
        self.alien_speed_factor=1
        self.feet_direction=1
        self.feet_drop_speed=20
    def update(self):
        self.alien_rect.x+=self.alien_speed_factor*self.feet_direction
    def plot_alien(self):
        self.screen.blit(self.image,self.alien_rect)
    def check_edges(self):
        screen_rect=self.screen.get_rect()
        if self.alien_rect.right>=screen_rect.right:
            return True
        elif self.alien_rect.left<=0:
            return True
        else:
            return False



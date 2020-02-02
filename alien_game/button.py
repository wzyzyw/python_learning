import pygame
class Button():
    def __init__(self,screen,msg):
        self.screen=screen
        self.screen_rect=self.screen.get_rect()
        #设置按钮的尺寸和其他属性
        self.width=200
        self.height=50
        self.text_color=(255,255,255)
        self.button_color=(0,255,0)
        self.font=pygame.font.SysFont(None,48)

        #创建按钮的rect对象，并居中
        self.rect=pygame.Rect(0,0,self.width,self.height)
        self.rect.center=self.screen_rect.center
        self.prep_msg(msg)

    def prep_msg(self,msg):
        """
        将msg渲染为图像，并在按钮上居中
        :param msg:
        :return:
        """
        self.msg_image=self.font.render(msg,True,self.text_color,self.button_color)
        self.msg_image_rect=self.msg_image.get_rect()
        self.msg_image_rect.center=self.rect.center

    def plot_button(self):
        #绘制一个颜色填充的按钮，再绘制文本
        self.screen.fill(self.button_color,self.rect)
        self.screen.blit(self.msg_image,self.msg_image_rect)

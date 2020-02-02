import pygame.font
class ScoreBoard():
    def __init__(self,game_setting,screen,stats):
        self.screen=screen
        self.screen_rect=screen.get_rect()
        self.game_setting=game_setting
        self.stats=stats

        #显示得分信息时使用的字体设置
        self.text_color=(30,30,30)
        self.font=pygame.font.SysFont(None,48)
        #准备初始得分图像
        self.prep_score()
        self.prep_high_score()


    def prep_score(self):
        #将得分转换为一幅渲染图像
        rounded_score=int(round(self.stats.score,-1))
        score_str="{:,}".format(rounded_score)
        self.score_imag=self.font.render(score_str,True,self.text_color,self.game_setting.bg_color)
        #将得分放在屏幕右上角
        self.score_rect=self.score_imag.get_rect()
        self.score_rect.right=self.screen_rect.right-20
        self.score_rect.top=20


    def prep_high_score(self):
        rounded_score = int(round(self.stats.highest_score, -1))
        score_str = "{:,}".format(rounded_score)
        self.high_score_imag = self.font.render(score_str, True, self.text_color, self.game_setting.bg_color)
        # 将得分放在屏幕右上角
        self.high_score_rect = self.high_score_imag.get_rect()
        self.high_score_rect.centerx = self.screen_rect.centerx
        self.high_score_rect.top = self.screen_rect.top


    def show_score(self):
        self.screen.blit(self.score_imag,self.score_rect)
        self.screen.blit(self.high_score_imag,self.high_score_rect)


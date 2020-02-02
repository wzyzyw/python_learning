import pygame
from setting import Setting
from ship import Ship
import game_function as gf
from alien import Alien
from gamestats import GameStats
from button import Button
from scoreboard import ScoreBoard


def run_game():
    #初始化游戏并创建一个屏幕对象
    pygame.init()
    #生成游戏场景对象，screen对象，飞船，外星人，子弹，游戏状态，按钮，记分牌
    game_setting=Setting()
    screen= pygame.display.set_mode((game_setting.screen_width,game_setting.screen_height))
    pygame.display.set_caption("Alien Invsion")
    bullets = []
    ship = Ship(screen)
    play_button = Button(screen, 'Play')
    stats=GameStats(game_setting)
    score_board=ScoreBoard(game_setting,screen,stats)
    aliens=[]
    gf.create_alien_feet(game_setting,screen,ship,aliens)
    #开始游戏的主循环
    while True:
        #获取事件响应
        gf.check_event(game_setting,screen,ship,bullets,stats,play_button,aliens)
        if stats.game_active:
            ship.update()
            gf.update_bullet(game_setting,screen,ship,bullets,aliens,score_board,stats)
            gf.update_alien(game_setting,screen,bullets,aliens,ship,stats)
        gf.update_screen(game_setting, screen, ship, bullets,aliens,play_button,stats,score_board)


if __name__=='__main__':
    run_game()








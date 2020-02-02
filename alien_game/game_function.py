import sys
import pygame
from bullet import Bullet
from alien import Alien
from time import sleep


def check_event(game_setting,screen,ship,bullets,stats,play_button,aliens):
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type==pygame.MOUSEBUTTONDOWN:
            mouse_x,mouse_y=pygame.mouse.get_pos()
            check_play_button(game_setting,screen,ship,aliens,bullets,stats,play_button,mouse_x,mouse_y)
        elif event.type==pygame.KEYDOWN:
            if event.key==pygame.K_RIGHT:
                ship.moving_right=True

            elif event.key==pygame.K_LEFT:
                ship.moving_left=True

            elif event.key==pygame.K_SPACE:
                if len(bullets)<game_setting.bullet_allow:
                    new_bullet=Bullet(screen,ship)
                    bullets.append(new_bullet)
            elif event.key==pygame.K_q:
                sys.exit()
        elif event.type==pygame.KEYUP:
            if event.key==pygame.K_RIGHT:
                ship.moving_right=False
            elif event.key==pygame.K_LEFT:
                ship.moving_left=False


def update_screen(game_setting,screen,ship,bullets,aliens,play_button,stats,score_board):
    # 更新屏幕,并用背景色填充屏幕
    screen.fill(game_setting.bg_color)
    ship.plot_ship()
    for bullet in bullets:
        bullet.plot_bullet()
    # print('screen alien',len(aliens))
    for alien in aliens:
        alien.plot_alien()

    score_board.show_score()
    #如果游戏处于非活动状态，就绘制Play按钮
    if not stats.game_active:
        play_button.plot_button()
    #屏幕可见
    pygame.display.flip()


def update_bullet(game_setting,screen,ship,bullets,aliens,score_board,stats):
    #判断子弹是否击中外星人
    collision_dict={}
    for bullet in bullets:
        bullet.update()
        for alien in aliens:
            bullet_collision_alien(alien, bullet, collision_dict)
    #字典里长度代表击中的外星人数量
    #统计得分
    len_dict=len(collision_dict)
    stats.score+=game_setting.alien_point*len_dict
    score_board.prep_score()
    check_high_score(stats, score_board)
    # 删除已经消失的子弹或者击中外星人的子弹
    collision_bullet=collision_dict.keys()
    count=0
    for bullet in bullets:
        if bullet in collision_bullet:
            del bullets[count]
        elif bullet.bullet_rect.top <= 0:
            del bullets[count]
        count+=1
    #删除被击中的外星人
    collision_alien=collision_dict.values()
    count=0
    for alien in aliens:
        if alien in collision_alien:
            del aliens[count]
        count+=1
    #如果外星人都被消灭了，产生新的外星人
    if len(aliens)==0:
        while bullets:
            del bullets[0]
        create_alien_feet(game_setting,screen,ship,aliens)
    # print('bullet len is ', len(bullets))


def update_alien(game_setting,screen,bullets,aliens,ship,stats):
    check_feet_edges(aliens)
    for alien in aliens:
        alien.update()
    alien_bottom=[alien.alien_rect.bottom for alien in aliens]
    max_bottom=max(alien_bottom)
    if max_bottom>ship.image_rect.top:
            ship_hit(game_setting,stats,screen,ship,aliens,bullets)


def get_num_alien_x(game_setting,alien_width):
    avaliable_space_x = game_setting.screen_width - 2 * alien_width
    num_alien_x = int(avaliable_space_x / (alien_width * 2))
    return num_alien_x


def create_aliens(screen,aliens,alien_num,row_num):
    alien=Alien(screen)
    alien_width=alien.alien_rect.width
    alien_height=alien.alien_rect.height
    tmp_x=alien_width+2*alien_width*alien_num
    tmp_y=alien_height+2*alien_height*row_num
    alien.alien_rect.x=tmp_x
    alien.alien_rect.y=tmp_y
    aliens.append(alien)


def get_num_alien_y(game_setting,alien_height,ship_height):
    avaliable_y=game_setting.screen_height-3*alien_height-ship_height
    num_alien_y=int(avaliable_y/(2*alien_height))
    return num_alien_y


def create_alien_feet(game_setting,screen,ship,aliens):
    alien = Alien(screen)
    alien_width = alien.alien_rect.width
    alien_height=alien.alien_rect.height
    ship_height=ship.image_rect.height
    num_alien_x = get_num_alien_x(game_setting, alien_width)
    num_alien_y= get_num_alien_y(game_setting, alien_height, ship_height)
    for i in range(num_alien_y):
        for j in range(num_alien_x):
            create_aliens(screen, aliens, j,i)


def check_feet_edges(aliens):
    for alien in aliens:
        if alien.check_edges():
            change_fleet_direction(aliens)
            break


def change_fleet_direction(aliens):
    """
    将整群外星人下移，并改变左右移动的方向
    :param aliens:
    :return:
    """
    for alien in aliens:
        alien.alien_rect.y+=alien.feet_drop_speed
        alien.feet_direction*=-1


def bullet_collision_alien(alien,bullet,collision_dict):
    if bullet.bullet_rect.top<=alien.alien_rect.bottom and (bullet.bullet_rect.top>alien.alien_rect.top):
        if (bullet.bullet_rect.left>alien.alien_rect.left) and (bullet.bullet_rect.right<alien.alien_rect.right):
            collision_dict[bullet]=alien


def ship_hit(game_setting,stats,screen,ship,aliens,bullets):
    if stats.remain_ship>0:
        stats.decrease_ship()
        #清空外星人和子弹列表
        while aliens:
            del aliens[0]
        while bullets:
            del bullets[0]
        #创建新的外星人群，并将飞船移到中央
        create_alien_feet(game_setting,screen,ship,aliens)
        ship.ship_center()
        sleep(0.5)
    else:
        stats.game_active=False
        pygame.mouse.set_visible(True)


def check_play_button(game_setting,screen,ship,aliens,bullets,stats,play_button,mouse_x,mouse_y):
    """
    当游戏处于非活跃状态且玩家单击play按钮时开始新的游戏
    :param stats:
    :param play_button:
    :param mouse_x:
    :param mouse_y:
    :return:
    """
    if (play_button.rect.collidepoint(mouse_x,mouse_y)) and (not stats.game_active):
        #重置游戏统计信息
        stats.game_active=True
        stats.reset()
        #清除外星人和子弹列表
        while aliens:
            del aliens[0]
        while bullets:
            del bullets[0]
        #创建新的外星人并让飞船居中
        create_alien_feet(game_setting,screen,ship,aliens)
        ship.ship_center()
        #隐藏光标
        pygame.mouse.set_visible(False)


def check_high_score(stats,score_board):
    if stats.score>stats.highest_score:
        stats.highest_score=stats.score
        with open(stats.path,'w') as file:
            file.write(str(stats.highest_score))
        score_board.prep_high_score()


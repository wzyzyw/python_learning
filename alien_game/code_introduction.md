# 程序结构

* 主函数：主函数结构如图所示

  ![main_function](https://github.com/wzyzyw/python_learning/tree/master/alien_game/image/main.png)

* Setting类：定义游戏场景参数，如游戏界面大小，背景色，玩家拥有飞船数

* GameStats类：定义游戏状态，活跃，非活跃，玩家剩余飞船数，游戏分数和历史最高分

* ScoreBoard类：显示历史最高分和游戏得分

* Ship类：定义飞船，读取飞船图像，飞船移动，更新飞船状态，在屏幕上绘制飞船

* Alien类：定义外星人，读取外星人图像，外星人移动，更新外星人状态，在屏幕上绘制外星人

* Bullet类：定义子弹，用rect表示，子弹移动，子弹飞船状态，在屏幕上绘制子弹

* Button类：定义按钮，开始游戏，重置游戏

* game_function函数模块：

    * check_event:获取鼠标、键盘事件，并进行处理如移动飞船，发射子弹，停止游戏，开始、重置游戏

    * update_screen：调用alien，ship等对象的绘图函数，将状态更新后的游戏元素画在屏幕上，并调用pygame的函数显示出来

    * update_bullet：

      ![update_bullet](https://github.com/wzyzyw/python_learning/blob/master/alien_game/image/update_bullet.png)

    * update_alien：判断外星人是否移动到左右边界，如果是更改左右移动方向，判断外星人是否与飞船碰撞，如果碰撞，清除外星人和子弹，并将飞船以得到中央，重新开始游戏

    * create_alien_feet：创建多个外星人，注意不越界，不重叠

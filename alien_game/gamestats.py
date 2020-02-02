import os
class GameStats():
    def __init__(self,gamesetting):
        self.remain_ship=gamesetting.ship_num
        self.all_ship=gamesetting.ship_num
        self.game_active=False
        self.score=0
        self.path="history_score/highest_score.txt"
        if os.path.exists(self.path):
            with open(self.path,'r') as file:
                tmp_score=file.read()
            self.highest_score=int(tmp_score)
        else:
            self.highest_score=0
            with open(self.path,'w') as file:
                file.write(str(self.highest_score))

    def decrease_ship(self):
        self.remain_ship-=1

    def reset(self):
        self.remain_ship=self.all_ship
        self.score=0


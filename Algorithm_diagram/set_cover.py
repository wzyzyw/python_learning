"""
算法图解
第8章贪婪算法解决集合覆盖问题
set与list类似，但不能出现重复元素,此外set还可以进行集合运算,差集，并集，交集
"""
def greed():
    remain_states=states_need
    best_stations=[]
    while remain_states:
        most_cover_station = None
        most_cover = 0
        for station,states_set in stations.items():
            cover=states_set & remain_states
            if len(cover)>most_cover:
                most_cover=len(cover)
                most_cover_station=station
        best_stations.append(most_cover_station)
        remain_states=remain_states-stations[most_cover_station]
        print('remain_states:',remain_states)
    print('cover all states')
    return best_stations





if __name__=='__main__':
    states_need=set(['mt','wa','or','id','nv','ut','ca','az'])
    stations={}
    stations['kone']=set(['id','nv','ut'])
    stations['ktwo']=set(['wa','id','mt'])
    stations['kthree']=set(['or','nv','ca'])
    stations['kfour']=set(['nv','ut'])
    stations['kfive']=set(['ca','az'])
    best_station=greed()
    for station in best_station:
        print('station:',station)
        print('cover states:',stations[station])

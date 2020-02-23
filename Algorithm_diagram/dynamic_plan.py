"""
算法图解第9章，9.2题动态规划解决背包问题
"""
import pandas as pd
import numpy as np


def dynamic_plan():
    need_goods=[['0']* (row*column)] #该列表存放每个单元格对应的物品，仅用一个下标表示
    last_line=pd.DataFrame(np.zeros((1,column)))#这个pandas表格用于存放上一行的值，初始化为0，便于处理更新第一行的情况
    last_line.rename(columns={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}, inplace=True)
    count=0
    for key,dict in goods.items():#迭代更新每一行
        weight=dict['weight']
        value=dict['value']
        for i in divide_capacity: #依次和每列的重量比较
            if weight<i:
                remain_weight=i-weight
                loc=divide_capacity.index(remain_weight)#找到前一行的该重量的位置
                new_value=last_line.at[0,divide_capacity[loc]]+value#当前价值+剩余价值
                if new_value>last_line.at[0,i]:
                    net.at[key,i]=new_value
                    if count>5:
                        need_goods[0][count]=need_goods[0][count-column-(count%column-loc)]+','+key #前面的物品加上当前物品
                    else:
                        need_goods[0][count] = key  # 正好是当前物品，count<5说明更新第一行
                else:
                    net.at[key,i]=last_line.at[0,i]
                    need_goods[0][count] = need_goods[0][count - column]  # 正好是前一行的同列
            elif weight==i:
                if value > last_line.at[0,i]:
                    net.at[key, i] = value
                    need_goods[0][count]=key # 正好是当前物品
                else:
                    net.at[key, i] = last_line.at[0,i]
                    need_goods[0][count] = need_goods[0][count - column]  # 正好是前一行的同列
            else:
                net.at[key, i] = last_line.at[0,i]
                if count>5:
                    need_goods[0][count]=need_goods[0][count-column]#正好是前一行的同列
                else:
                    need_goods[0][count]='0'
            count += 1
        print(net)
        last_line.loc[0]=net.loc[key]
    print('need goods is:')
    print(need_goods[0][count-1])
        # tmp.rename(index={key:0},inplace=True)
        # last_line[0]=tmp


if __name__=="__main__":
    row=5#网格行数
    column=6#网格列数
    capacity=6#背包总容量
    net_array=np.zeros((row,column))
    net=pd.DataFrame(net_array) #创建pandas数据
    items=['water','book','food','jack','camera']
    divide_capacity=[i+1 for i in range(column)]
    net.index=items #修改pandas表格的行名
    net.rename(columns={0:1,1:2,2:3,3:4,4:5,5:6},inplace=True)#修改pandas表格的列名

    goods={'water':{'weight':3,'value':10},'book':{'weight':1,'value':3},
           'food':{'weight':2,'value':9},
           'jack':{'weight':2,'value':5},
           'camera':{'weight':1,'value':6}} #建立物品字典，每样物品都有重量和价值属性

    dynamic_plan()#进行动态规划求解


"""
算法图解一书中，第6章广度优先python实现
实现中使用到队列结构，一度关系中，如果没有找到，将该一度关系的邻居都加入到队列
后面，在顺序查找下一个一度关系，直到一度关系查完，队列中只剩下二度关系
（1）对于有增有减的序列，队列适合
（2）算法和结构匹配
（3）有的节点即是 一度节点也是二度节点的邻居，很容易使代码陷入死循环
因此，需要列表标记已经查找的节点
"""
from collections import deque
def bfs_search():
    graph={}
    graph['you']=['alice','bob','claire']
    graph['bob']=['anuj','peggy']
    graph['alice']=['peggy']
    graph['claire']=['thom','jonny']
    graph['anuj']=[]
    graph['peggy']=[]
    graph['thom']=[]
    graph['jonny']=[]

    search_que=deque()
    search_que+=graph['you']
    searched=[]#这个数组用于记录检查过的人
    search_destination='thom'
    while search_que:
         person=search_que.popleft()#取出队列中的第一个人
         if  person not in searched:
             searched.append(person)
             if person==search_destination:
                 print(person+" is a mango seller")
                 return True
             else:
                 search_que+=graph[person]
    return False

if __name__=='__main__':
    bfs_search()


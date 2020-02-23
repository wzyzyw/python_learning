"""
算法图解，第7章，dijkstra算法实现
注意点
（1）三个字典：每个点的邻居及权重，cost，parent，实例的代码初始化了三个字典，但对于多节点而言，需要改进
（2）嵌套字典，添加新的键值对的语法
（3）float(inf),或者import math 后，直接使用inf

"""

def dijkstra():
    graph={}
    # graph['start']={'A',6,'B',2}
    # graph['A']={'terminal',1}
    # graph['B']={'terminal',5}
    graph["start"]={}
    graph['start']['A']=6
    graph['start']['B'] = 2
    graph['A']={}
    graph['A']['terminal'] = 1
    graph['B']={}
    graph['B']['terminal'] = 1 #嵌套字典的正确表达式
    graph['terminal']={}
    cost={}
    cost['A']=6
    cost['B']=2
    cost['terminal']=float('inf')
    parent={}
    parent['A']='start'
    parent['B']='start'
    parent['terminal']=None
    processed=[]
    node=find_lowest_cost_node(cost, processed)

    while node is not None:
        processed.append(node)
        current_neighbor=graph[node]
        for key in current_neighbor.keys():
            new_cost=cost[node]+current_neighbor[key]
            if cost[key]>new_cost:
                cost[key] = new_cost
                parent[key]=node
        node=find_lowest_cost_node(cost,processed)

    parent_node=parent['terminal']
    lowest_path=[parent_node,'terminal']
    while parent_node is not 'start':
        parent_node=parent[parent_node]
        lowest_path.insert(0,parent_node)
    print('the lowest path from start to terminal is:\n')
    print(lowest_path)


def find_lowest_cost_node(cost,processed):
    lowest=float('inf')
    lowest_cost_node=None
    for node in cost.keys():
        if (cost[node]<lowest) and(node not in processed):
            lowest=cost[node]
            lowest_cost_node=node

    return lowest_cost_node


if __name__=='__main__':
    dijkstra()

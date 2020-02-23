"""
python implement sorting algorithm
"""
import numpy as np
def quick_sortting(array):

    len_arr=len(array)
    # 基线数组递归的基线条件是元素为1或者为0，
    # 有时候这两种情况等效的，用<2比较方便，
    # 考虑有时候会划分出空list，也归于此类
    if len_arr<2:
        return array
    else:
        base_index=np.random.randint(0,len_arr,1)
        base_point=array.pop(base_index[0])
        less_area=[]
        higher_area=[]
        for x in array:
            if x<base_point:
                less_area.append(x)
            else:
                higher_area.append(x)
        #list拼接，单个元素用[]表明是元素为1的列表
        return quick_sortting(less_area)+[base_point]+quick_sortting(higher_area)





if __name__=='__main__':
    length=5
    array=[]
    index=0
    while index<length:
        tmp=np.random.randint(0,length*3,1)
        if tmp[0] in array:
            continue
        else:
            array.append(tmp[0])
            index+=1
    print('original array:', array)
    quick_sort_array=quick_sortting(array)
    print('quick sorting array:',quick_sort_array)





import pandas as pd
import numpy as np


# 将每个用户的启动日期存入list里,然后去重、排序，算出连续登录天数的最大值
def Continuous_launch_Count(arr):
    ls = []
    it = iter(arr)
    for value in it:
        # print(value)
        ls.append(value)
    ls.sort()
    ls = list(set(ls))
    maxcount=1
    count=1
    if(len(ls)>1):
        for i in range(len(ls)-1):
            if(ls[i]+1==ls[i+1]):
                count=count+1
                if count>=maxcount :
                    maxcount=count
            else:
                count=1
    else:
        count=1
    return maxcount

#该方法用于添加最长连续登陆天数这一特征
def Add_continuous_launch_Feature(temp_launch):
    launch_continuous_count = temp_launch.groupby('user_id').agg(Continuous_launch_Count).reset_index()
    launch_continuous_count.rename(columns={'app_launch': 'launch_continuous_count'}, inplace=True)
    #print(launch_continuous_count[launch_continuous_count['user_id']==14808])
    return launch_continuous_count
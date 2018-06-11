import pandas as pd
import numpy as np

#这个函数根据app启动的日期，算出间隔序列，存在interval_list里,对于只启动一次的用户，置为0
def Interval_list(arr):
    list=[]
    interval_list=[]
    #将每个用户的启动日期存入list里
    it=iter(arr)
    for value in it:
        #print(value)
        list.append(value)
    list.sort()
    if len(list)>1:
        for i in range(len(list)-1):
            interval_list.append(list[i+1]-list[i])
    else:
        interval_list.append(0)
    return interval_list

#计算启动间隔的mean
def Interval_Mean(ser):
    list=Interval_list(ser)
    return np.mean(list)

#计算启动间隔的最大值
def Interval_Max(ser):
    list=Interval_list(ser)
    return np.max(list)
#计算启动间隔的最小值
def Interval_Min(ser):
    list=Interval_list(ser)
    return np.min(list)
#计算启动间隔的方差
def Interval_Var(ser):
    list=Interval_list(ser)
    return np.var(list)

#计算启动间隔的cv
def Interval_CV(ser):
    list=Interval_list(ser)
    return np.std(list)/np.mean(list)

#temp_launch = pd.read_table('app_launch_log.txt',names = ['user_id','app_launch'])#,seq = '\t')
# 时间分片以及对应的特征抽取

# 用户登录的时间间隔的特征
def Add_launch_Interval_Feature(temp_launch):
    #按user_id统计app启动次数
    #launch_res = temp_launch.groupby('user_id').count().reset_index()
    #print(launch_res.head())
    #统计app启动间隔的平均值
    launch_interval_mean=temp_launch.groupby('user_id').agg(Interval_Mean).reset_index().fillna(0)
    launch_interval_mean.rename(columns={'app_launch': 'launch_interval_mean'}, inplace=True)
    launch_res=pd.merge(temp_launch,launch_interval_mean,on = 'user_id',how = 'left')

    # 统计app启动间隔的最大值
    launch_interval_max = temp_launch.groupby('user_id').agg(Interval_Max).reset_index().fillna(0)
    launch_interval_max.rename(columns={'app_launch': 'launch_interval_max'}, inplace=True)
    launch_res = pd.merge(launch_res, launch_interval_max, on='user_id', how='left')

    # 统计app启动间隔的最小值
    launch_interval_min = temp_launch.groupby('user_id').agg(Interval_Min).reset_index().fillna(0)
    launch_interval_min.rename(columns={'app_launch': 'launch_interval_min'}, inplace=True)
    launch_res = pd.merge(launch_res, launch_interval_min, on='user_id', how='left')

    # 统计app启动间隔的方差
    launch_interval_var = temp_launch.groupby('user_id').agg(Interval_Var).reset_index().fillna(0)
    launch_interval_var.rename(columns={'app_launch': 'launch_interval_var'}, inplace=True)
    launch_res = pd.merge(launch_res, launch_interval_var, on='user_id', how='left')

    # 统计app启动间隔的CV
    launch_interval_cv = temp_launch.groupby('user_id').agg(Interval_CV).reset_index().fillna(0)
    launch_interval_cv.rename(columns={'app_launch': 'launch_interval_cv'}, inplace=True)
    launch_res = pd.merge(launch_res, launch_interval_cv, on='user_id', how='left')

    #print(temp_launch[temp_launch['user_id']==105])
    #launch_interval_max=temp_launch.groupby('user_id').agg().reset_index()
    del launch_res['app_launch']
    launch_res=launch_res.drop_duplicates()
    return launch_res

#该函数用于添加用户创建视频的平均天数
def Add_create_Interval_Feature(temp_video):
    create_interval_mean = temp_video.groupby('user_id').agg(Interval_Mean).reset_index().fillna(0)
    create_interval_mean.rename(columns={'video_create': 'create_interval_mean'}, inplace=True)
    #create_interval_mean.drop_duplicates()
    #print(create_interval_mean[create_interval_mean['user_id']==107685])
    return create_interval_mean



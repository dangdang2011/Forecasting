# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection

#对每个用户的action，构建特征分别是Max,Min,Max_type,Min_type,Median,Std,Skew,Kurt
def Max_action_count(x):
    return x.max()
def Min_action_count(x):
    return x.min()
#返回action 的类型，取值从0到5
def Max_action_type(x):
    return float(x.idxmax()[-1])
#返回action 的类型，取值从0到5
def Min_action_type(x):
    return float(x.idxmin()[-1])

def Mean_action(x):
    return x.mean()
def Median_action(x):
    return x.median()
def Std_action(x):
    return x.std()
def Skew_action(x):
    return x.skew()
def Kurt_action(x):
    return x.kurt()

def AddFeature(train_feature):
    # train_feature=pd.DataFrame(train_feature[(train_feature['action_type_0']>0) |\
    #                             (train_feature['action_type_1']>0 )|\
    #                              (train_feature['action_type_2']>0 )| \
    #                             (train_feature['action_type_3'] > 0)|\
    #                              (train_feature['action_type_4']>0 )|\
    #                              (train_feature['action_type_5']>0) ])
    # 选取出action0--actoin5,统计出max_action

    max_action_count = pd.DataFrame(train_feature[['action_type_0', 'action_type_1', 'action_type_2',\
                                                   'action_type_3', 'action_type_4', 'action_type_5']]\
                                    ).apply(Max_action_count, axis=1)

    # 将每个user的max_action加入到train_feature中
    train_feature = pd.concat([train_feature, max_action_count], axis=1, join='inner')
    # 将新加入的feature命名为max_action_count
    train_feature.rename(columns={0: 'max_action_count'}, inplace=True)

    # 统计出现次数最少的action
    min_action_count = pd.DataFrame(train_feature[['action_type_0', 'action_type_1', 'action_type_2',\
                                                   'action_type_3', 'action_type_4', 'action_type_5']]\
                                    ).apply(Min_action_count, axis=1)
    train_feature = pd.concat([train_feature, min_action_count], axis=1, join='inner')
    train_feature.rename(columns={0: 'min_action_count'}, inplace=True)

    # # # 出现次数最多的action类型
    # max_action_type = pd.DataFrame(train_feature.iloc[:, [5, 6, 7, 8, 9, 10]]).apply(Max_action_type, axis=1)
    # train_feature = pd.concat([train_feature, max_action_type], axis=1, join='inner')
    # train_feature.rename(columns={0: 'max_action_type'}, inplace=True)
    #
    # # # 出现次数最少的action类型
    # min_action_type = pd.DataFrame(train_feature.iloc[:, [5, 6, 7, 8, 9, 10]]).apply(Min_action_type, axis=1)
    # train_feature = pd.concat([train_feature, min_action_type], axis=1, join='inner')
    # train_feature.rename(columns={0: 'min_action_type'}, inplace=True)

    # action的均值
    mean_action =pd.DataFrame(train_feature[['action_type_0', 'action_type_1', 'action_type_2',\
                                             'action_type_3', 'action_type_4', 'action_type_5']]\
                             ).apply(Mean_action, axis=1)
    train_feature = pd.concat([train_feature, mean_action], axis=1, join='inner')
    train_feature.rename(columns={0: 'mean_action'}, inplace=True)

    # action的中位数
    median_action = pd.DataFrame(train_feature[['action_type_0', 'action_type_1', 'action_type_2',\
                                                'action_type_3', 'action_type_4', 'action_type_5']]\
                                ).apply(Median_action, axis=1)
    train_feature = pd.concat([train_feature, median_action], axis=1, join='inner')
    train_feature.rename(columns={0: 'median_action'}, inplace=True)

    # action的标准差
    std_action = pd.DataFrame(train_feature[['action_type_0', 'action_type_1', 'action_type_2',\
                                                'action_type_3', 'action_type_4', 'action_type_5']]\
                                ).apply(Std_action, axis=1)
    train_feature = pd.concat([train_feature, std_action], axis=1, join='inner')
    train_feature.rename(columns={0: 'std_action'}, inplace=True)

    # action的三阶矩
    skew_action = pd.DataFrame(train_feature[['action_type_0', 'action_type_1', 'action_type_2',\
                                                 'action_type_3', 'action_type_4', 'action_type_5']]\
                                 ).apply(Skew_action, axis=1)
    train_feature = pd.concat([train_feature, skew_action], axis=1, join='inner')
    train_feature.rename(columns={0: 'skew_action'}, inplace=True)

    # action的四阶矩
    kurt_action = pd.DataFrame(train_feature[['action_type_0', 'action_type_1', 'action_type_2',\
                                                 'action_type_3', 'action_type_4', 'action_type_5']]\
                                 ).apply(Kurt_action, axis=1)
    train_feature = pd.concat([train_feature, kurt_action], axis=1, join='inner')
    train_feature.rename(columns={0: 'kurt_action'}, inplace=True)

    # print train_feature.head()
    return train_feature

def Add_day_action(train_feature,opendate,close_date):
    list=[]
    x=2
    list.append(x)
    for i in range (0,close_date-opendate):
        x=x+6
        list.append(x)
    for j in range (0,6):
        max_day_action=pd.DataFrame(train_feature.iloc[:,list]).apply(Max_action_count,axis=1)
        train_feature = pd.concat([train_feature, max_day_action], axis=1, join='inner')
        train_feature.rename(columns={0: 'max_day_action_'+str(j)}, inplace=True)

        min_day_action = pd.DataFrame(train_feature.iloc[:, list]).apply(Min_action_count, axis=1)
        train_feature = pd.concat([train_feature, min_day_action], axis=1, join='inner')
        train_feature.rename(columns={0: 'min_day_action_' + str(j)}, inplace=True)

        mean_day_action = pd.DataFrame(train_feature.iloc[:, list]).apply(Mean_action, axis=1)
        train_feature = pd.concat([train_feature, mean_day_action], axis=1, join='inner')
        train_feature.rename(columns={0: 'mean_day_action_' + str(j)}, inplace=True)

        median_day_action = pd.DataFrame(train_feature.iloc[:, list]).apply(Median_action, axis=1)
        train_feature = pd.concat([train_feature, median_day_action], axis=1, join='inner')
        train_feature.rename(columns={0: 'median_day_action_' + str(j)}, inplace=True)

        sdt_day_action = pd.DataFrame(train_feature.iloc[:, list]).apply(Std_action, axis=1)
        train_feature = pd.concat([train_feature, sdt_day_action], axis=1, join='inner')
        train_feature.rename(columns={0: 'sdt_day_action_' + str(j)}, inplace=True)

        skew_day_action = pd.DataFrame(train_feature.iloc[:, list]).apply(Skew_action, axis=1)
        train_feature = pd.concat([train_feature, skew_day_action], axis=1, join='inner')
        train_feature.rename(columns={0: 'skew_day_action_' + str(j)}, inplace=True)

        kurt_day_action = pd.DataFrame(train_feature.iloc[:, list]).apply(Kurt_action, axis=1)
        train_feature = pd.concat([train_feature, kurt_day_action], axis=1, join='inner')
        train_feature.rename(columns={0: 'kurt_day_action_' + str(j)}, inplace=True)

        for i in range(len(list)):
            list[i] = list[i] + 1
    list=[x for x in range(2,(close_date-opendate+1)*6+2)]
    train_feature.drop(train_feature.columns[list], axis=1, inplace=True)
    print('test')
    print(train_feature.info())
    #print(train_feature.iloc[0:5,[2, 8, 14, 20]])
    #print(train_feature.iloc[0:5, [ 26, 32, 38, 44]])
    #print(train_feature[['max_day_action_0','max_day_action_1']].head())

    return train_feature
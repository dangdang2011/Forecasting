import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection

# file = pd.read_csv('pre20180601.csv',names = ['user_id','action1','action2','action3','action4','action5','action6','launch_times','create_times'])

# print(file)
launch = pd.DataFrame(pd.read_table('app_launch_log.txt',names = ['user_id','app_launch']))#,seq = '\t'
register = pd.DataFrame(pd.read_table('user_register_log.txt',names = ['user_id','register_day','register_type','device type']))
video = pd.DataFrame(pd.read_table('video_create_log.txt',names = ['user_id','video_create']))
activity = pd.DataFrame(pd.read_table('user_activity_log.txt',names = ['user_id','day_times','page','video_id','author_id'
    ,'action_type']))

# 时间分片以及对应的抽取
def timeInterval(type,startdate,enddate):
    temp_launch = launch[(launch['app_launch'] >= startdate) & (launch['app_launch'] <= enddate)]
    temp_register = register[(register['register_day'] >= startdate) & (register['register_day'] <= enddate)]
    temp_video = video[(video['video_create'] >= startdate) & (video['video_create'] <= enddate)]
    temp_activity = activity[(activity['day_times'] >= startdate) & (activity['day_times'] <= enddate)]


    # 特征抽取
    activity_res = pd.DataFrame(temp_activity.groupby(['user_id', 'action_type'])[['day_times']].count().unstack().fillna(0))
    launch_res = temp_launch.groupby('user_id').count()
    video_res = temp_video.groupby('user_id').count()

    feature = temp_register.join(launch_res.join(activity_res)).join(video_res).fillna(0)
    # feature = pd.concat([temp_register,activity_res,launch_res,video_res])
    print(feature)

    return feature

train_feature = timeInterval('train',1,23)

train_id = train_feature['user_id']

# test_feature = timeInterval('test',24,30)
# test_id = test_feature['user_id']
#
# print("train_id",train_id)
# print("test_id",test_id)
#
# train_label = []
# for i in range(len(train_id)):
#     if i in test_id:
#         train_label.append(1)
#     else:
#         train_label.append(0)
# print(train_label)
#
#
# # 评分准则
# def get_score(pre,true):
#     # result = []
#     # index = 0
#     count = 0 # count表示预测结果与真实结果的并集。
#     # print(type(pre))
#     # print(type(true))
#     for index in range(len(pre)):
#         if(pre[index] in true):# 说明预测对了
#             count += 1
#     precision = count / len(pre)
#     recall = count / len(true)
#     print("count",count)
#     f1_score = (2 * precision * recall) / (precision + recall)
#     return f1_score
#
# # 模型选择
# from sklearn.ensemble import GradientBoostingClassifier
#
# used_feature = [4,5,6,7,8,9,10,11]
# # 验证训练集
# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature.iloc[:,used_feature], train_label, test_size=0.3,random_state=1017)
#
# # Y_train = train_label
# # X_train = train_feature
# # X_test = test_feature
# gbdt = GradientBoostingClassifier()
# gbdt.fit(X_train,Y_train)
#
# predict = gbdt.predict(X_test)
# true_data = Y_test
# print(get_score(predict,true_data))
#
# # 验证测试集
# X_train = train_feature.iloc[:,used_feature]
# Y_train = train_label
#
#
# predict = gbdt.predict(X_train)
# result = []
# print(len(predict))
#
# for i in range(len(predict)):
#     if(predict[i] == 1):
#         print(train_feature.iloc[i, [0]])
#         # result.append()
# print(result)
#
# validation = test_feature.iloc[:,[0]]
# print(get_score(predict,validation))






# feature = pd.merge(activity_res,launch_res, on = 'user_id')

# if (type == 'train'):
#     # feature = launch_res.join(video.join(activity_res))#,on = 'user_id',how = 'left')
#     # feature = temp_register.join(feature)#,on = 'user_id',how = 'left')
#     # # feature = feature.join(video_res)#,on = 'user_id',how = 'left').fillna(0)
#
#     feature = launch_res.join(activity_res)  # .set_index('user_id').join(activity_res.set_index('user_id'))
#     feature = temp_register.set_index('user_id').join(feature)
#     # result = result.drop(['register_day', 'register_type', 'device type'], axis=1)
#     feature = feature.join(video_res).fillna(0)
#     # print(feature)
# else:
#     feature = launch_res.join(activity_res.join(video_res))  # ,on = 'user_id',how = 'left')
#     # feature = feature.join(video_res.set_index('user_id'))#,on = 'user_id',how = 'left').fillna(0)
# print(feature)
# # feature = feature.join(video_res)
# # feature = feature.join(temp_register)
# # print(feature)
# # feature = pd.concat([temp_register,activity_res,launch_res,video_res])
# # feature = temp_register.join(activity_res.join(launch_res)).join(video_res).fillna(0)
# # else:
# #     activity_res = pd.DataFrame(temp_activity.groupby(['user_id', 'action_type'])[['day_times']].count().unstack().fillna(0))
# #     launch_res = temp_launch.groupby('user_id').count()
# #     video_res = temp_video.groupby('user_id').count()
# #     feature = activity_res.join(launch_res).join(video_res).fillna(0)
#
# # feature = .fillna(0)#.join(activity_res.join(launch_res)).join(video_res).fillna(0))
#
# # else:# 构造测试集
# #     activity_res = pd.DataFrame(temp_activity.groupby(['user_id', 'action_type'])[['day_times']].count().unstack().fillna(0))
# #     launch_res = temp_launch.groupby('user_id').count()
# #     video_res = temp_video.groupby('user_id').count()
# #     feature = pd.concat()
# # print(type,"\n",feature)
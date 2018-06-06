import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection

# file = pd.read_csv('pre20180601.csv',names = ['user_id','action1','action2','action3','action4','action5','action6','launch_times','create_times'])

# 读取源数据
launch = pd.DataFrame(pd.read_table('app_launch_log.txt',names = ['user_id','app_launch'],index_col = 0))#,seq = '\t'
register = pd.DataFrame(pd.read_table('user_register_log.txt',names = ['user_id','register_day','register_type','device type']))
video = pd.DataFrame(pd.read_table('video_create_log.txt',names = ['user_id','video_create']))
activity = pd.DataFrame(pd.read_table('user_activity_log.txt',names = ['user_id','day_times','page','video_id','author_id'
    ,'action_type']))

# 时间分片以及对应的特征抽取
def timeInterval(startdate,enddate):
    temp_launch = launch[(launch['app_launch'] >= startdate) & (launch['app_launch'] <= enddate)]
    temp_register = register[(register['register_day'] >= startdate) & (register['register_day'] <= enddate)]
    temp_video = video[(video['video_create'] >= startdate) & (video['video_create'] <= enddate)]
    temp_activity = activity[(activity['day_times'] >= startdate) & (activity['day_times'] <= enddate)]
    # temp_activity.to_csv('temp_activity.csv')

    # 特征抽取
    activity_res = pd.DataFrame(temp_activity['day_times'].groupby([temp_activity['user_id'],temp_activity['action_type']]).count().unstack().fillna(0))
    launch_res = temp_launch.groupby('user_id').count()
    video_res = temp_video.groupby('user_id').count()
    activity_res = pd.DataFrame(temp_activity['day_times'].groupby(
        [temp_activity['user_id'], temp_activity['action_type']]).count().unstack().fillna(0))
    launch_res = temp_launch.groupby('user_id').count()
    video_res = temp_video.groupby('user_id').count()

    # feature = temp_register.join(launch_res.join(activity_res)).join(video_res).fillna(0)
    feature = pd.concat([temp_register,activity_res,launch_res,video_res],axis=1).fillna(0)

    # print(feature)
    return feature


def testInterval(startdate,enddate):
    temp_launch = launch[(launch['app_launch'] >= startdate) & (launch['app_launch'] <= enddate)]
    # temp_register = register[(register['register_day'] >= startdate) & (register['register_day'] <= enddate)]
    temp_video = video[(video['video_create'] >= startdate) & (video['video_create'] <= enddate)]
    temp_activity = activity[(activity['day_times'] >= startdate) & (activity['day_times'] <= enddate)]
    temp_register = register[(register['register_day'] >= 1) & (register['register_day'] <= startdate)]

    activity_res = pd.DataFrame(temp_activity['day_times'].groupby(
        [temp_activity['user_id'], temp_activity['action_type']]).count().unstack().fillna(0))
    launch_res = pd.DataFrame(temp_launch.groupby('user_id').count())
    video_res = pd.DataFrame(temp_video.groupby('user_id').count())

    # print(len(launch_res),len(video_res),len(activity_res))

    # feature = pd.concat([activity_res,launch_res,video_res],axis=0).fillna(0)# 默认是并集
    feature = pd.concat([temp_launch,temp_video,temp_activity]).fillna(0)
    print(feature.index)#index绝对不对


    return feature['user_id']

#选择1-23日数据作为训练集
train_feature = timeInterval(1,23)
#抽取训练的用户id
train_id = train_feature['user_id']
# print(train_id)
# 抽取测试集的特征以及测试集内的user_id
# test_feature =
test_id = testInterval(24,30)

# print(test_id)

# print("train_id",train_id)
# print("test_id",test_id)

# 打标签，如果1-23日注册的用户在24-30日出现过活动，则视为活跃用户
train_label = []
for i in range(len(train_id)):
    if i in test_id:
        train_label.append(1)

    else:
        train_label.append(0)
# print(train_label)

# 评分准则函数
def get_score(pre,true):
    # result = []
    # index = 0
    count = 0 # count表示预测结果与真实结果的并集。
    print(len(pre))
    print(len(true))
    for index in range(len(pre)):
        if(pre[index] in true):# 说明预测对了
            # print(pre[index])
            count += 1

    precision = count / len(pre) # 计算公式
    recall = count / len(true) # 计算公式
    print("count",count)

    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score

# 模型选择
from sklearn.ensemble import GradientBoostingClassifier

# 我们先选择8种特征进行计算
used_feature = [4,5,6,7,8,9,10,11]

# 验证模型用的划分，将训练集（1-23日）按test_size划分为测试集和验证集，用来验证模型本身的好坏。
# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature.iloc[:,used_feature], train_label, test_size=0.3,random_state=1017)

# Y_train = train_label
# X_train = train_feature
# X_test = test_feature


# 将train_feature作为新的测试集用来作为模型的输入
train_feature = train_feature.iloc[:,used_feature]
# 建模
gbdt = GradientBoostingClassifier()
#拟合模型
gbdt.fit(train_feature,train_label)

#预测label
predict = gbdt.predict(train_feature)

# #验证模型的好坏，使用评分准则
# true_data = Y_test
# # print(get_score(predict,true_data))

# # label为已经打好的label
# Y_train = train_label
#
# # 用模型预测1-23日的输出结果，标号为1即为活跃用户
# predict = gbdt.predict(train_feature)

# 输出所有结果
result = []
# print(len(train_id),len(train_feature))
for i in range(len(predict)):
    if(predict[i] == 1):
        result.append(train_id.iloc[i])
        # print(train_id.iloc[i]) # 输出搞好啦！

print(get_score(result,test_id))

result = pd.DataFrame(result)
result.to_csv('result.csv')


# print(test_id)
# print(result)

        # result.append()
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
# 特征抽取
# activity_res = pd.DataFrame(temp_activity['day_times'].groupby([temp_activity['user_id'],temp_activity['action_type']]).count().unstack().fillna(0))
# # activity_res = pd.DataFrame(
# #     temp_activity.groupby(['user_id', 'action_type'])[['day_times']].count().unstack().fillna(0))
# # print(len(activity_res))
# # print(activity_res)
# # activity_res.to_csv('activity_res.csv')
# launch_res = temp_launch.groupby('user_id').count()
# video_res = temp_video.groupby('user_id').count()

# print(activity_res)
# print(launch_res)
# print(video_res)
# feature = launch_res.join(activity_res).join(video_res).fillna(0)
# print(temp_activity)
# print(temp_activity.index)

# print(len(activity_res))
# print(activity_res)
# activity_res.to_csv('activity_res.csv')
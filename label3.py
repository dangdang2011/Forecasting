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

    # 特征抽取
    activity_res = pd.DataFrame(temp_activity['day_times'].groupby(
        [temp_activity['user_id'], temp_activity['action_type']]).count().unstack().fillna(0))
    launch_res = temp_launch.groupby('user_id').count()
    video_res = temp_video.groupby('user_id').count()

    feature = temp_register.join(launch_res.join(activity_res)).join(video_res).fillna(0)

    return feature

def testInterval(startdate,enddate):
    temp_launch = launch[(launch['app_launch'] >= startdate) & (launch['app_launch'] <= enddate)]
    # temp_register = register[(register['register_day'] >= startdate) & (register['register_day'] <= enddate)]
    temp_video = video[(video['video_create'] >= startdate) & (video['video_create'] <= enddate)]
    temp_activity = activity[(activity['day_times'] >= startdate) & (activity['day_times'] <= enddate)]
    temp_register = register[(register['register_day'] >= 1) & (register['register_day'] <= startdate)]

    # activity_res = pd.DataFrame(temp_activity['day_times'].groupby(
    #     [temp_activity['user_id'], temp_activity['action_type']]).count().unstack().fillna(0))
    # launch_res = pd.DataFrame(temp_launch.groupby('user_id').count())
    # video_res = pd.DataFrame(temp_video.groupby('user_id').count())

    # feature = pd.concat([activity_res,launch_res,video_res],axis=0).fillna(0)# 默认是并集
    feature = pd.concat([temp_launch,temp_video,temp_activity]).fillna(0)
    # print(feature.index)#index绝对不对


    return feature['user_id']

#选择1-23日数据作为训练集
train_feature = timeInterval(1,23)
#抽取训练的用户id
train_id = train_feature['user_id']
# print(train_id)
# 抽取测试集的特征以及测试集内的user_id
# test_feature =
test_id = testInterval(24,30)

# 打标签，如果1-23日注册的用户在24-30日出现过活动，则视为活跃用户
train_label = []
for i in range(len(train_id)):
    if i in test_id:
        train_label.append(1)

    else:
        train_label.append(0)

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

# 将train_feature作为新的测试集用来作为模型的输入
train_feature = train_feature.iloc[:,used_feature]
# 建模
gbdt = GradientBoostingClassifier()
#拟合模型
gbdt.fit(train_feature,train_label)

#预测label
predict = gbdt.predict(train_feature)

# 输出所有结果
result = []
# print(len(train_id),len(train_feature))
for i in range(len(predict)):
    if(predict[i] == 1):
        result.append(train_id.iloc[i])
        # print(train_id.iloc[i]) # 输出搞好啦！

print(get_score(result,test_id))
result = pd.DataFrame(result)

# 最终结果
train_feature = timeInterval(1,30)
train_id = train_feature['user_id']
train_feature = train_feature.iloc[:,used_feature]

result = []
predict = gbdt.predict(train_feature)
print(len(predict))
for i in range(len(predict)):
    if(predict[i] == 1):
        result.append(train_id.iloc[i])
        # print(train_id.iloc[i]) # 输出搞好啦！

result = pd.DataFrame(result)
print(result)
# print(result.groupby())

result.to_csv('result.csv',index=None)

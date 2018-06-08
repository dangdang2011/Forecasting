# -*- coding: UTF-8 -*
import pandas as pd
import numpy as np
import addFeature as af
from sklearn import preprocessing, model_selection

# 读取源数据
launch = pd.read_table('app_launch_log.txt',names = ['user_id','app_launch'])#,seq = '\t')
register = pd.read_table('user_register_log.txt',names = ['user_id','register_day','register_type','device type'])
video = pd.read_table('video_create_log.txt',names = ['user_id','video_create'])
activity = pd.read_table('user_activity_log.txt',names = ['user_id','day_times','page','video_id','author_id'
    ,'action_type'])

# 时间分片以及对应的特征抽取
def trainInterval(startdate,enddate):
    temp_launch = launch[(launch['app_launch'] >= startdate) & (launch['app_launch'] <= enddate)]
    temp_register = register[(register['register_day'] >= startdate) & (register['register_day'] <= enddate)]
    temp_video = video[(video['video_create'] >= startdate) & (video['video_create'] <= enddate)]
    temp_activity = activity[(activity['day_times'] >= startdate) & (activity['day_times'] <= enddate)]

    activity_res = temp_activity.groupby(['user_id','action_type'])['day_times'].size().unstack().fillna(0).reset_index()
    # print(activity_res.names())
    launch_res = temp_launch.groupby('user_id').count().reset_index()
    video_res = temp_video.groupby('user_id').count().reset_index()

    feature = pd.merge(launch_res,activity_res,on = 'user_id',how = 'left')
    feature = pd.merge(feature,video_res,on = 'user_id',how='left')
    feature = pd.merge(temp_register,feature,on='user_id',how='left')
    feature = feature.fillna(0)
    feature.rename(columns={0: 'action_type_0', 1: 'action_type_1', 2: 'action_type_2',\
                            3: 'action_type_3', 4: 'action_type_4', 5: 'action_type_5'}, inplace = True)
    feature.to_csv('feature.csv')
    return feature

def testInterval(startdate,enddate):
    temp_launch = launch[(launch['app_launch'] >= startdate) & (launch['app_launch'] <= enddate)]
    temp_video = video[(video['video_create'] >= startdate) & (video['video_create'] <= enddate)]
    temp_activity = activity[(activity['day_times'] >= startdate) & (activity['day_times'] <= enddate)]
    feature = pd.concat([temp_launch, temp_video, temp_activity])
    user_id = np.unique(feature['user_id'])#.drop_duplicates()
    return user_id

#获取1-23日的用户特征数据
train_feature = trainInterval(1,23)

#这里要引入addFeature模块，增加的特征是Max,Min,Max_action_type,Min_action_type,Median,Std,Skew,Kurt
train_feature = af.AddFeature(train_feature)

#提取id
train_id = train_feature['user_id']
#获取24-30日产生数据的用户id
test_id = testInterval(24,30)

# 打标签，如果1-23日注册的用户在24-30日出现过活动，则视为活跃用户
active = 0
train_label = []
true_user = []
for item in train_id:
    if item in test_id:
        train_label.append(1)
        true_user.append(item)
        active+= 1
    else:
        train_label.append(0)
print"1-23日活跃用户有：",active

# 评分准则函数
def get_score(pre,true):
    count = 0 # count表示预测结果与真实结果的并集。
    print("预测数据有：",len(pre))
    print("真实数据有：",len(true))
    for index in range(len(pre)):
        if(pre[index] in true):# 说明预测对了
            # print(pre[index])
            count += 1
    precision = count / len(pre) # 计算公式
    recall = count / len(true) # 计算公式
    print"precision is:",precision,"recall is:",recall
    f1_score = (2 * precision * recall) / (precision + recall)
    print"正阳例，即我们预测结果中，真正的活跃用户有：",count,"评分：",f1_score

from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# 我们先选择几种特征进行计算
used_feature = [4,5,6,7,8,9,10,11]

# 将used_feature 作为待选特征，赋值给train_set，作为训练集输入模型
train_set = train_feature.iloc[:,used_feature]

# 建模，20180607测试（最下面的函数测试的）GBDT最优秀，准确率0.72左右，所以用gbdt提交
gbdt = GradientBoostingClassifier()
gbdt.fit(train_set,train_label)

# 提交,这里文件的名字都称为final
final_feature = trainInterval(1,30)
final_id = final_feature['user_id']
final_set = final_feature.iloc[:,used_feature]

result = []
predict = gbdt.predict(final_set)
print"最终预测了",len(predict),"条数据"
for i in range(len(predict)):
    if(predict[i] == 1):
        result.append(final_id.iloc[i])
print "其中，最终提交数据：",len(result),"条"
result = pd.DataFrame(result)
result.to_csv('result.csv',index=None)

# 20180607，今天，最好的模型是GBDT
# !!!注意，这部分是用来验证模型好坏的，最终提交的部分里，模型暂时写死
# 这个函数定义了拟合模型，可以选择不同的模型来拟合，用来快速测试
def modelUse(model_name,data_set,data_label,data_id):
    model = model_name
    # 在这里我们选择model_Set中的一种model进行拟合,用data_set 和data_label来拟合
    model.fit(data_set, data_label)
    # 基于上面的模型，我们给出预测结果
    predict = model.predict(data_set)
    # 输出所有结果
    result = []
    for i in range(len(predict)):
        if(predict[i] == 1):# 1 表示真正的活跃用户
            result.append(data_id.iloc[i])
            # print(train_id.iloc[i]) # 输出搞好啦！
    #给出模型评分
    get_score(result,true_user)

# 下面注释掉的语句在测试模型的时候用
# model_Set = [XGBClassifier(),GradientBoostingClassifier()]
#
# for model in model_Set:
#     print("在这里使用了模型：",model)
#     modelUse(model_name=model,data_set=train_set,data_id=train_id,data_label=train_label)







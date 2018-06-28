import pandas as pd
import numpy as np
import addFeature as af
from sklearn import preprocessing, model_selection,metrics
from sklearn.decomposition import PCA
import AddIntervalFeature as aif
import AddContinuousFeature as acf
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from lightgbm import LGBMClassifier
import lightgbm as lgb
import datetime
warnings.filterwarnings('ignore')

# 读取源数据
launch = pd.read_table('app_launch_log.txt',names = ['user_id','app_launch'])#,seq = '\t')
register = pd.read_table('user_register_log.txt',names = ['user_id','register_day','register_type','device_type'])
video = pd.read_table('video_create_log.txt',names = ['user_id','video_create'])
activity = pd.read_table('user_activity_log.txt',names = ['user_id','day_times','page','video_id','author_id'
    ,'action_type'])

def slice(opendate,closedate):# 去特征的区间划分

    temp_launch = launch[(launch['app_launch'] >= opendate) & (launch['app_launch'] <= closedate)]
    temp_video = video[(video['video_create'] >= opendate) & (video['video_create'] <= closedate)]
    temp_activity = activity[(activity['day_times'] >= opendate) & (activity['day_times'] <= closedate)]
    # 按groupby取特征：action的数量和page，launch次数，video create次数
    activity_res = temp_activity.groupby(['user_id', 'action_type'])['day_times'].size().unstack().fillna(0).reset_index()
    activity_page = temp_activity.groupby(['user_id', 'page'])['day_times'].size().unstack().fillna(0).reset_index()
    launch_res = temp_launch.groupby('user_id').count().reset_index()
    video_res = temp_video.groupby('user_id').count().reset_index()
    #这里统计数author_id被播放、关注、点赞、转发、举报和减少推荐的数量
    author_res = temp_activity.groupby(['author_id', 'action_type'])['day_times'].size().unstack().fillna(0).reset_index()

    #activity在第一天没有action_type==4的行为，所以如果只取这天的话，要手动加上这列，置为0
    if 4 not in activity_res.columns:
        activity_res[4]=0.0

    # 这里就是给列改名了
    author_res.rename(columns={0: 'au_action_type_0', 1: 'au_action_type_1', 2: 'au_action_type_2', \
                               3: 'au_action_type_3', 4: 'au_action_type_4', 5: 'au_action_type_5'}, inplace=True)
    activity_res.rename(columns={0: 'action_type_0', 1: 'action_type_1', 2: 'action_type_2', \
                                 3: 'action_type_3', 4: 'action_type_4', 5: 'action_type_5'}, inplace=True)
    # print(activity_res.head())
    activity_page.rename(columns={0: 'action_page_0', 1: 'action_page_1', 2: 'action_page_2', \
                                  3: 'action_page_3', 4: 'action_page_4'}, inplace=True)
    feature = pd.merge(launch_res, activity_res, on='user_id', how='left')
    feature = pd.merge(feature, activity_page, on='user_id', how='left')
    feature = pd.merge(feature, video_res, on='user_id', how='left').fillna(0)# 补充没产生行为的用户，标记为0

    #将具有user_id==author_id的用户合并，如果author_id不在左边的表中，则忽略
    feature = pd.merge(feature, author_res, left_on='user_id', right_on='author_id', how='left').fillna(0)
    del feature['author_id']
    # print(feature[feature['user_id'] == 3197][['user_id', 'app_launch']])
    # print("Feature",feature.info())
    feature = af.AddFeature(feature)

    launch_interval = aif.Add_launch_Interval_Feature(temp_launch)
    create_interval=aif.Add_create_Interval_Feature(temp_video) # 这里增加了平均创作视频间隔
    launch_continuous = acf.Add_continuous_launch_Feature(temp_launch) # 这里增加了连续登陆的最大天数

    feature=pd.merge(feature,create_interval,on='user_id',how='left').fillna(0)
    feature = pd.merge(feature, launch_interval, on='user_id', how='left')
    feature = pd.merge(feature, launch_continuous, on='user_id', how='left').fillna(0)

    used_feature = [ i for i in range(1, feature.columns.size)] # 0 为user_id 不能作为特征。这里由于没有注册信息，所以从第一个特征开始选择。
    # print(used_feature)
    used_set = feature.iloc[:, used_feature]

    # PCA特征抽取，经检验PCA可以提高本地精度。
    pca = PCA(n_components=4, copy=True, whiten=False)
    used_set_pca = pca.fit_transform(used_set) # 使用PCA对标准化后的特征进行降维
    print('单个变量方差贡献率', pca.explained_variance_ratio_)
    PCA_feature = pd.DataFrame(used_set_pca)
    PCA_feature['user_id'] = feature['user_id']
    feature = pd.merge(feature, PCA_feature)
    # print(feature)
    return feature

# 时间分片以及对应的特征抽取
def trainInterval(startdate, boundarydate, enddate,w1,w2):     # boundarydate用于划分时间区间,现在我们划分两个区间，以16号为分界。
    temp_register = register[(register['register_day'] >= startdate) & (register['register_day'] <= enddate)]
    # weight = [3,7] # 权值list

    #获取第一区间特征并加权
    first_feature = slice(startdate,boundarydate)
    # print(first_feature.head())
    used_feature = [i for i in range(1, first_feature.columns.size)]
    first_feature.iloc[:, used_feature] = first_feature.iloc[:, used_feature] * w1
    # print(first_feature.head())

    #获取第二区间特征并加权 如果有first则boubdarydate 必须+1 否则会重复计算boundarydate这一天的数据
    second_feature = slice(boundarydate+1,enddate)
    used_feature = [i for i in range(1, second_feature.columns.size)]
    second_feature.iloc[:, used_feature] = second_feature.iloc[:, used_feature] * w2
    # print(second_feature.head())

    second_feature.add(first_feature)
    # print(second_feature.head())
    # 构造特征集合
    feature = pd.merge(temp_register,second_feature, on='user_id', how='left').fillna(0)
    # sns.heatmap(feature.corr(), annot=True, annot_kws={'size':8}, cmap="RdYlGn", xticklabels=True, yticklabels=True,linewidths=0)
    # ax = plt.gca()
    # for label in ax.xaxis.get_ticklabels():
    #     label.set_rotation(45)
    # for label in ax.yaxis.get_ticklabels():
    #     label.set_rotation(0)
    # ax.invert_yaxis()
    # plt.show()
    return feature

def testInterval(startdate,enddate):
    temp_launch = launch[(launch['app_launch'] >= startdate) & (launch['app_launch'] <= enddate)]
    temp_video = video[(video['video_create'] >= startdate) & (video['video_create'] <= enddate)]
    temp_activity = activity[(activity['day_times'] >= startdate) & (activity['day_times'] <= enddate)]

    feature = pd.concat([temp_launch, temp_video, temp_activity])
    user_id = np.unique(feature['user_id'])#.drop_duplicates()
    return user_id

data_1=trainInterval(1,7,14,w1=0.5,w2=0.5)
train_id_1=data_1['user_id']
test_id_1=testInterval(15,22)

data_2=trainInterval(8,14,21,w1 = 0.5,w2 = 0.5)
train_id_2=data_2['user_id']
test_id_2=testInterval(22,29)

# train_feature = trainInterval(1,1,23) #提train的特征
# train_id = train_feature['user_id'] #提取id
# test_id = testInterval(24,30) #获取24-30日产生数据的用户id

#print("Feature",train_feature.info())

# 评分准则函数
def get_score(pre,true):
    count = 0 # count表示预测结果与真实结果的并集。
    print("预测数据有：",len(pre),"真实数据有：",len(true))
    for index in range(len(pre)):
        if(pre[index] in true):# 说明预测对了
            # print(pre[index])
            count += 1
    precision = count / len(pre) # 计算公式
    recall = count / len(true) # 计算公式
    print("precision is:",precision,"recall is:",recall)
    f1_score = (2 * precision * recall) / (precision + recall)
    print("正阳例，即我们预测结果中，真正的活跃用户有：",count,"评分：",f1_score)


# 打标签，如果1-23日注册的用户在24-30日出现过活动，则视为活跃用户
def MakeLabel(train_id,test_id):
    active = 0
    deactive = 0
    train_label = []
    true_user = []
    for item in train_id:
        if item in test_id:
            train_label.append(1)
            true_user.append(item)
            active += 1
        else:
            train_label.append(0)
            deactive += 1
    print("活跃用户有：", active, "人")
    print("不活跃用户有：", deactive, "人")
    return train_label,true_user

#给data1和data2的数据都打上标签
train_label_1 = []
true_user_1 = []
train_label_1,true_user_1=MakeLabel(train_id_1,test_id_1)

train_label_2 = []
true_user_2 = []
train_label_2,true_user_2=MakeLabel(train_id_2,test_id_2)

#提取出data1的特征和data2的特征，放在train_set_data1和train_set_data2中
#print('feature',data_1.info())
used_feature=[i for i in range(4, data_1.columns.size)] # 将used_feature 作为待选特征，赋值给train_set，作为训练集输入模型
train_set_1 = data_1.iloc[:,used_feature]
train_set_2 = data_2.iloc[:,used_feature]

#装载lgb的训练数据，data1和标签
#装载lgbd的验证数据，data2和标签
train_data=lgb.Dataset(train_set_1,label=train_label_1)
lgb_eval =lgb.Dataset(train_set_2,label=train_label_2)

#调参
# params={
#     #以下参数提高准确率
#     'learning_rate':[0.01,0.05,0.1,0.15,0.2,0.25,0.3],
#     'max_depth':[3,4,5,6,7,8],
#     'num_leaves':[31,50,70,90,110,127,150,170,200,220,250],
#     #'num_iterations':[50,100,150,200,250,300,400,500],
#     #'max_bin':[x for x in range(100,255,5)],
#     # 'min_data_in_leaf':[x for x in range(20,200,5)],
#     #'n_estimators':[x for x in range(10,500,10)]
# }
# lgbm=lgb.LGBMClassifier()
# grid=GridSearchCV(lgbm,params,cv=5,scoring='f1')
# #开始调参
# print('开始调参')
# grid.fit(train_set_2,train_label_2)
# print('最佳分数',grid.best_score_)
# print('最佳参数',grid.best_params_)
# print('最佳模型',grid.best_estimator_)

# # lightgbm初始参数
param_test = {
    'boosting_type':'rf',
    'num_leaves':200,
    'objective':'binary',
    'max_depth':5,
    'learning_rate':0.1,
    'n_estimators':100,
    'feature_fraction': 0.9, # 建树的特征选择比例
    'bagging_fraction': 0.8, # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

#
# lightgbm开始训练模型
# lgbm=lgb.train(params=param_test,train_set=train_data)
# #用data2测试，计算评分
# predict=lgbm.predict(train_set_2,num_iteration=lgbm.best_iteration)
#试试二分类器的效果
lgbm=lgb.LGBMClassifier(
    boosting_type='gbdt',
    num_leaves=31,
    objective='binary',
    max_depth=5,
    learning_rate=0.01,
    n_estimators=100,
    feature_fraction= 0.9, # 建树的特征选择比例
    bagging_fraction=0.8, # 建树的样本采样比例
    bagging_freq= 5,  # k 意味着每 k 次迭代执行bagging
    verbose= 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    )
lgbm.fit(train_set_1,train_label_1)
predict=lgbm.predict(train_set_2)
result = []
for i in range(len(predict)):
    if (predict[i] == 1):  # 阈值大于0.42表示真正的活跃用户
        result.append(train_id_2.iloc[i])
        # print(train_id.iloc[i]) # 输出搞好啦！
# 给出模型评分
# print("使用真实数据的结果")
get_score(result, true_user_2)

#将data1和data2合并作为新的训练集，用上面确定的参数，训练出新模型
final_trainset=train_set_1.append(train_set_2)
final_train_label=train_label_1+train_label_2
final_train_data=lgb.Dataset(final_trainset,final_train_label)
print('开始训练模型')
#final_model=lgb.train(params=param_test,train_set=final_train_data)
final_model=lgbm.fit(final_trainset,final_train_label)


print('开始预测数据')
#提交部分
final_feature = trainInterval(1,15,30,0,1)
final_id = final_feature['user_id']
final_set = final_feature.iloc[:,used_feature]
result = []
predict = final_model.predict(final_set)
print("最终预测 了",len(predict),"条数据")
for i in range(len(predict)):
    if(predict[i] ==1 ):
        result.append(final_id.iloc[i])
print("其中，最终提交数据：",len(result),"条")
result = pd.DataFrame(result)
result.to_csv('result-lgb.csv',index=None)
import pandas as pd
import numpy as np
import addFeature as af
from sklearn import preprocessing, model_selection
from sklearn.decomposition import PCA
import AddIntervalFeature as aif
import AddContinuousFeature as acf
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

# 读取源数据
launch = pd.read_table('app_launch_log.txt',names = ['user_id','app_launch'])#,seq = '\t')
register = pd.read_table('user_register_log.txt',names = ['user_id','register_day','register_type','device type'])
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

    # 这里就是给列改名了
    activity_res.rename(columns={0: 'action_type_0', 1: 'action_type_1', 2: 'action_type_2', \
                                 3: 'action_type_3', 4: 'action_type_4', 5: 'action_type_5'}, inplace=True)
    # print(activity_res.head())
    activity_page.rename(columns={0: 'action_page_0', 1: 'action_page_1', 2: 'action_page_2', \
                                  3: 'action_page_3', 4: 'action_page_4'}, inplace=True)
    # print(activity_res.names())

    feature = pd.merge(launch_res, activity_res, on='user_id', how='left')
    feature = pd.merge(feature, activity_page, on='user_id', how='left')
    feature = pd.merge(feature, video_res, on='user_id', how='left').fillna(0)# 补充没产生行为的用户，标记为0

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
def trainInterval(startdate, boundarydate, enddate):     # boundarydate用于划分时间区间,现在我们划分两个区间，以16号为分界。
    #注册时间这里我需要和你讨论一下
    temp_register = register[(register['register_day'] >= startdate) & (register['register_day'] <= enddate)]
    weight = [0,1] # 权值list

    #获取第一区间特征并加权
    first_feature = slice(startdate,boundarydate-1)
    # print(first_feature.head())
    used_feature = [i for i in range(1, first_feature.columns.size)]
    first_feature.iloc[:, used_feature] = first_feature.iloc[:, used_feature] * weight[0]
    # print(first_feature.head())

    #获取第二区间特征并加权
    second_feature = slice(boundarydate,enddate)
    used_feature = [i for i in range(1, second_feature.columns.size)]
    second_feature.iloc[:, used_feature] = second_feature.iloc[:, used_feature] * weight[1]

    second_feature.add(first_feature)

    # 构造特征集合
    feature = pd.merge(temp_register,second_feature, on='user_id', how='left').fillna(0)
    # feature.fillna(0)
    # print(feature)

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

train_feature = trainInterval(1, 16 ,23) #提train的特征
train_id = train_feature['user_id'] #提取id
test_id = testInterval(24,30) #获取24-30日产生数据的用户id

# 打标签，如果1-23日注册的用户在24-30日出现过活动，则视为活跃用户
active = 0
deactive = 0
train_label = []
true_user = []
for item in train_id:
    if item in test_id:
        train_label.append(1)
        true_user.append(item)
        active+= 1
    else:
        train_label.append(0)
        deactive+=1
print("活跃用户有：",active,"人")
print("不活跃用户有：",deactive,"人")

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

from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn import svm

from sklearn.linear_model import LogisticRegression

used_feature=[i for i in range(4, train_feature.columns.size)] # 将used_feature 作为待选特征，赋值给train_set，作为训练集输入模型
train_set = train_feature.iloc[:,used_feature]

# !!!注意，这部分是用来验证模型好坏的，最终提交的部分里，模型暂时写死
# #这个函数定义了拟合模型，可以选择不同的模型来拟合，用来快速测试
def modelUse(model_name,data_set,data_label,data_id):
    # 切分训练
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data_set, data_label, test_size=0.3, random_state=1)
    model = model_name
    # 在这里我们选择model_Set中的一种model进行拟合,用data_set 和data_label来拟合
    print(X_train)
    model.fit(X_train, Y_train)
    # 基于上面的模型，我们给出预测结果
    predict = model.predict(data_set)
    # 输出所有结果
    result = []
    for i in range(len(predict)):
        if(predict[i] == 1):# 1 表示真正的活跃用户
            result.append(data_id.iloc[i])
            # print(train_id.iloc[i]) # 输出搞好啦！
    #给出模型评分
    # print("使用真实数据的结果")
    get_score(result,true_user)


# 下面注释掉的语句在测试模型的时候用
model_Set = [XGBClassifier(),GradientBoostingClassifier(),LogisticRegression()]

for model in model_Set:
    print("在这里使用了模型：",model)
    modelUse(model_name=model,data_set=train_set,data_id=train_id,data_label=train_label)

## 模型调参

# def modelPara(data_train,data_label):
#     param_xg_test = {
#         # 'max_depth':[7,8,9,10],
#         # 'min_child_weight':[5,6,7,8                                                 ],
#         # 'gamma': [i / 10.0 for i in range(2, 4)],
#         'learning_rage':[i/10 for i in range(1,3)]
#     }
#
#     xgb = XGBClassifier(
#         learning_rate=0.1,
#         n_estimators=1000,
#         max_depth=5,
#         min_child_weight=1,
#         gamma=0,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective='binary:logistic',
#         nthread=4,
#         scale_pos_weight=1,
#         seed=27
#     )
#     param_gbdt_test = {
#     }
#     # estimator = GradientBoostingClassifier(min_sam)
#
#     # gsearch = GridSearchCV(estimator, param_grid=param_xg_test, scoring='f1', cv=5)
#     # gsearch.fit(data_train, data_label)
#     # print("Parameter:",gsearch.best_params_,gsearch.best_score_)
#
# modelPara(train_set,train_label)

##

#####
#
# 建模，201806013测试（最下面的函数测试的）GBDT最优秀，准确率0.8015左右，所以用gbdt提交
# gbdt = GradientBoostingClassifier()
# gbdt.fit(train_set,train_label)
# # 提交,这里文件的名字都称为final
# final_feature = trainInterval(1,30)
# final_id = final_feature['user_id']
# final_set = final_feature.iloc[:,used_feature]
# result = []
# predict = gbdt.predict(final_set)
# print("最终预测了",len(predict),"条数据")
# for i in range(len(predict)):
#     if(predict[i] == 1):
#         result.append(final_id.iloc[i])
# print("其中，最终提交数据：",len(result),"条")
# result = pd.DataFrame(result)
# result.to_csv('result.csv',index=None)

#####

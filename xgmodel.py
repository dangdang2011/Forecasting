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
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

def probafunc(proba_value,threshold):
    proba_value[proba_value >= threshold] = 1
    proba_value[proba_value < threshold] = 0
    return proba_value

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

# 打data_1的标签，如果1-23日注册的用户在24-30日出现过活动，则视为活跃用户
def label(train_id,test_id):
    active = 0; deactive = 0
    train_label = [] ; true_user = []
    for item in train_id:
        if item in test_id:
            train_label.append(1)
            true_user.append(item)
            active+= 1
        else:
            train_label.append(0)
            deactive+=1
    print("活跃用户有：",active,"人,不活跃用户有：",deactive,"人")
    return train_label,true_user

# 读取源数据
launch = pd.read_table('app_launch_log.txt',names = ['user_id','app_launch'])#,seq = '\t')
register = pd.read_table('user_register_log.txt',names = ['user_id','register_day','register_type','device type'])
video = pd.read_table('video_create_log.txt',names = ['user_id','video_create'])
activity = pd.read_table('user_activity_log.txt',names = ['user_id','day_times','page','video_id','author_id','action_type'])

def slice(opendate,closedate):# 特征的区间划分

    temp_launch = launch[(launch['app_launch'] >= opendate) & (launch['app_launch'] <= closedate)]
    temp_video = video[(video['video_create'] >= opendate) & (video['video_create'] <= closedate)]
    temp_activity = activity[(activity['day_times'] >= opendate) & (activity['day_times'] <= closedate)]
    # 按groupby取特征：action的数量和page，launch次数，video create次数
    activity_res = temp_activity.groupby(['user_id', 'action_type'])['day_times'].size().unstack().fillna(0).reset_index()
    activity_page = temp_activity.groupby(['user_id', 'page'])['day_times'].size().unstack().fillna(0).reset_index()
    launch_res = temp_launch.groupby('user_id').count().reset_index()
    video_res = temp_video.groupby('user_id').count().reset_index()
    #activity在第一天没有action_type==4的行为，所以如果只取这天的话，要手动加上这列，置为0
    if 4 not in activity_res.columns:
        activity_res[4]=0.0

    # 改列名
    activity_res.rename(columns={0: 'action_type_0', 1: 'action_type_1', 2: 'action_type_2', \
                                 3: 'action_type_3', 4: 'action_type_4', 5: 'action_type_5'}, inplace=True)
    # print(activity_res.head())
    activity_page.rename(columns={0: 'action_page_0', 1: 'action_page_1', 2: 'action_page_2', \
                                  3: 'action_page_3', 4: 'action_page_4'}, inplace=True)
    feature = pd.merge(launch_res, activity_res, on='user_id', how='left')
    feature = pd.merge(feature, activity_page, on='user_id', how='left')
    feature = pd.merge(feature, video_res, on='user_id', how='left').fillna(0)# 补充没产生行为的用户，标记为0

    feature = af.AddFeature(feature)

    launch_interval = aif.Add_launch_Interval_Feature(temp_launch)
    create_interval = aif.Add_create_Interval_Feature(temp_video) # 这里增加了平均创作视频间隔
    launch_continuous = acf.Add_continuous_launch_Feature(temp_launch) # 这里增加了连续登陆的最大天数

    feature = pd.merge(feature, create_interval,on='user_id',how='left').fillna(0)
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

    return feature

def testInterval(startdate,enddate):
    temp_launch = launch[(launch['app_launch'] >= startdate) & (launch['app_launch'] <= enddate)]
    temp_video = video[(video['video_create'] >= startdate) & (video['video_create'] <= enddate)]
    temp_activity = activity[(activity['day_times'] >= startdate) & (activity['day_times'] <= enddate)]

    feature = pd.concat([temp_launch, temp_video, temp_activity])
    user_id = np.unique(feature['user_id'])#.drop_duplicates()
    return user_id


data_1=trainInterval(1,8,15,w1=0.5,w2=0.5)
train_id_1=data_1['user_id']
test_id_1=testInterval(16,23)
# 抽取用户标签和真实活跃用户的标签
label_1, true_user_1 = label(train_id_1,test_id_1)
used_feature=[i for i in range(4, data_1.columns.size)] # 将used_feature 作为待选特征，赋值给train_set，作为训练集输入模型
data_set_1 = data_1.iloc[:,used_feature] #

data_2=trainInterval(8,15,22,w1 = 0.5,w2 = 0.5)
train_id_2=data_2['user_id']
test_id_2=testInterval(23,30)
label_2, true_user_2 = label(train_id_2,test_id_2)
used_feature = [i for i in range(4, data_2.columns.size)] # 将used_feature 作为待选特征，赋值给train_set，作为训练集输入模型
data_set_2 = data_2.iloc[:,used_feature] # 用data2验证模型的好坏，并重新（使用data1）调参。

# 切分训练、测试集，这个部分用来交叉验证。20180621版本先不用，因为我们用data1训练，用data2验证
# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_set, train_label, test_size=0.3, random_state=1)
model = XGBClassifier(
        # silent=0,  # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        # nthread=4,# cpu 线程数 默认最大
        learning_rate=0.1,  # 如同学习率
        min_child_weight=1,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        max_depth=4,  # 构建树的深度，越大越容易过拟合
        gamma=0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        subsample=1,  # 随机采样训练样本 训练实例的子采样比
        max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
        colsample_bytree=1,  # 生成树时进行的列采样
        reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        # reg_alpha=0, # L1 正则项参数
        # scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
        # objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
        # num_class=10, # 类别数，多分类与 multisoftmax 并用
        n_estimators=100,  # 树的个数
        seed=1000  # 随机种子
        # eval_metric= 'auc'
)

# 训练集使用data1
model.fit(data_set_1, label_1)

# 基于上面的模型，我们给出预测结果
# predict = model.predict(data_set_2)
proba = model.predict_proba(data_set_2)[:,1]
predict = probafunc(proba,threshold=0.42)

print(predict)
print(proba)

# 输出所有结果
result = []
for i in range(len(predict)):
    if(predict[i] == 1): #1 表示真正的活跃用户
        result.append(train_id_2.iloc[i])
#给出模型评分
get_score(result,true_user_2)

# data3 = trainInterval(1,14,21,0.3,0.7)

# 最后一次训练模型，用上面的参数
train_feature = data_1.append(data_2)
train_id = train_id_1.append(train_id_2)
test_id = np.append(test_id_1,test_id_2)
label, true_user = label(train_id,test_id)

train_set = train_feature.iloc[:,used_feature]
model.fit(train_set,label)

# # 提交,这里文件的名字都称为final
final_feature = trainInterval(1,15,30,0.5,0.5)
final_id = final_feature['user_id']
final_set = final_feature.iloc[:,used_feature]
predict = model.predict(final_set)
print("最终预测了",len(predict),"条数据")
result = []
for i in range(len(predict)):
    if(predict[i] == 1):
        result.append(final_id.iloc[i])
print("其中，最终提交数据：",len(result),"条")
result = pd.DataFrame(result)
result.to_csv('result.csv',index=None)

#
# #####

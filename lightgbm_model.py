import pandas as pd
import numpy as np
import addFeature as af
from sklearn import preprocessing, model_selection,metrics
from sklearn.decomposition import PCA
import AddIntervalFeature as aif
import AddContinuousFeature as acf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from lightgbm import LGBMClassifier
import lightgbm as lgb
import datetime
import matplotlib.pyplot as plt
from featureSelectiontest import trainInterval
from featureSelection import testInterval

warnings.filterwarnings('ignore')

# 读取源数据
launch = pd.read_table('app_launch_log.txt',names = ['user_id','app_launch'])#,seq = '\t')
register = pd.read_table('user_register_log.txt',names = ['user_id','register_day','register_type','device_type'])
video = pd.read_table('video_create_log.txt',names = ['user_id','video_create'])
activity = pd.read_table('user_activity_log.txt',names = ['user_id','day_times','page','video_id','author_id'
    ,'action_type'])


data_1=pd.read_csv('data1.csv')
data_2 = pd.read_csv('data2.csv')

train_id_1=data_1['user_id']
test_id_1=testInterval(17,23)

train_id_2=data_2['user_id']
test_id_2=testInterval(24,30)

# train_feature = trainInterval(1,1,23) #提train的特征
# train_id = train_feature['user_id'] #提取id
# test_id = testInterval(24,30) #获取24-30日产生数据的用户id

#print("Feature",train_feature.info())


from featureSelection import label,get_score

#给data1和data2的数据都打上标签
train_label_1 = []
true_user_1 = []
train_label_1,true_user_1=label(train_id_1,test_id_1)

train_label_2 = []
true_user_2 = []
train_label_2,true_user_2=label(train_id_2,test_id_2)

#提取出data1的特征和data2的特征，放在train_set_data1和train_set_data2中
#print('feature',data_1.info())
used_feature=[i for i in range(4, data_1.columns.size)] # 将used_feature 作为待选特征，赋值给train_set，作为训练集输入模型
train_set_1 = data_1.iloc[:,used_feature]
train_set_2 = data_2.iloc[:,used_feature]

#装载lgb的训练数据，data1和标签
#装载lgbd的验证数据，data2和标签
train_data=lgb.Dataset(train_set_1,label=train_label_1)
lgb_eval =lgb.Dataset(train_set_2,label=train_label_2)

#画图
# param_range = [2,4,6,8]
# train_scores, test_scores = validation_curve(
#     LGBMClassifier(),X=train_set_1,y=train_label_1,param_name='max_depth',param_range=param_range,cv = 10,scoring="accuracy",n_jobs=1
# )
#
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
#
# plt.plot(param_range, train_mean,
#          color='blue', marker='o',
#          markersize=5, label='training accuracy')
#
# plt.fill_between(param_range,
#                  train_mean + train_std,
#                  train_mean - train_std,
#                  alpha=0.15, color='blue')
#
# plt.plot(param_range, test_mean,
#          color='green', linestyle='--',
#          marker='s', markersize=5,
#          label='validation accuracy')
#
# plt.fill_between(param_range,
#                  test_mean + test_std,
#                  test_mean - test_std,
#                  alpha=0.15, color='green')
#
# plt.grid()
# plt.xlabel('Number of training samples')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# # plt.ylim([0.8, 1.0])
# # plt.tight_layout()
# plt.show()


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
    max_depth=4,
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
final_feature = pd.read_csv('final_feature.csv')
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
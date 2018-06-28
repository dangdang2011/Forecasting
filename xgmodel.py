import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
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

data_1=pd.read_csv('data_1.csv')
data_2 = pd.read_csv('data_2.csv')

from featureSelection import testInterval

train_id_1 = data_1['user_id']
test_id_1=testInterval(17,23)
# 抽取用户标签和真实活跃用户的标签
label_1, true_user_1 = label(train_id_1,test_id_1)
used_feature=[i for i in range(4, data_1.columns.size)] # 将used_feature 作为待选特征，赋值给train_set，作为训练集输入模型
data_set_1 = data_1.iloc[:,used_feature] #
param_range = [2,4,6,8]
train_scores, test_scores = validation_curve(
    XGBClassifier(),X=data_set_1,y=label_1,param_name='max_depth',param_range=param_range,cv = 10,scoring="accuracy",n_jobs=1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(param_range,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
# plt.ylim([0.8, 1.0])
# plt.tight_layout()
plt.show()
# print(train_scores,test_scores)
# print(train_scores.mean())
# print(test_scores.mean())
# print(train_scores.std())
# print(test_scores.std())
# # plt.grid()
# # plt.fill_between(param_range, train_scores.mean() - train_scores.std(), color = 'r', alpha = 0.1)
# # plt.fill_between(param_range, test_scores.mean() - test_scores.std(), color = 'g', alpha = 0.1)
# # plt.plot(param_range, train_scores.mean(),color = 'r')
# # plt.plot(param_range, train_scores.mean(),color = 'g')
# # plt.show()

train_id_2=data_2['user_id']
test_id_2 = testInterval(24,30)
label_2, true_user_2 = label(train_id_2,test_id_2)
data_set_2 = data_2.iloc[:,used_feature] # 用data2验证模型的好坏，并重新（使用data1）调参。

# 切分训练、测试集，这个部分用来交叉验证。20180621版本先不用，因为我们用data1训练，用data2验证
# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_set, train_label, test_size=0.3, random_state=1)
model = XGBClassifier(
        # silent=0,  # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        nthread=4,# cpu 线程数 默认最大
        learning_rate=0.1,  # 如同学习率
        min_child_weight=1,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        max_depth=5,  # 构建树的深度，越大越容易过拟合
        gamma=0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        # 'lambda' = 2, # 正则化
        subsample=0.7,  # 随机采样训练样本 训练实例的子采样比
        colsample_bytree=1,  # 生成树时进行的列采样
        max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
        reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        reg_alpha=0, # L1 正则项参数
        # scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
        # objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
        # num_class=10, # 类别数，多分类与 multisoftmax 并用
        n_estimators=100,  # 树的个数
        seed=1000  # 随机种子
        # eval_metric= 'auc'
)

## 模型调参
# def modelPara(data_train,data_label):
#     param_xg_test = {
#         # 'max_depth':[7,8,9,10],
#         # 'min_child_weight':[5,6,7,8],
#         # 'gamma': [i / 10.0 for i in range(2, 4)],
#         # 'learning_rage':[i/10 for i in range(1,3)]
#     }
#
#     xgb = XGBClassifier(
#         # max_depth=5,
#         # min_child_weight=1,
#         # gamma=0,
#         learning_rate=0.1,
#         n_estimators=1000,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective='binary:logistic',
#         nthread=4,
#         scale_pos_weight=1,
#         seed=27
#     )
#     param_gbdt_test = {
#     }
#     estimator = GradientBoostingClassifier()
#
#     gsearch = GridSearchCV(estimator, param_grid=param_xg_test, scoring='f1', cv=5)
#     gsearch.fit(data_train, data_label)
#     print("Parameter:",gsearch.best_params_,gsearch.best_score_)
#
#
# modelPara(data_set_1,data_label=label_1)



# 训练集使用data1
model.fit(data_set_1, label_1)

# 基于上面的模型，我们给出预测结果
# predict = model.predict(data_set_1)

proba = model.predict_proba(data_set_1)[:,1]
predict = probafunc(proba,threshold=0.42)
print("data1验证结果",predict)
print(proba)
# 输出所有结果
result = []
for i in range(len(predict)):
    if(predict[i] == 1): #1 表示真正的活跃用户
        result.append(train_id_1.iloc[i])
#给出模型评分
get_score(result,true_user_1)


proba = model.predict_proba(data_set_2)[:,1]
predict = probafunc(proba,threshold=0.42)
print("data_2验证结果",predict)
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
final_feature = trainInterval(1,15,30,0,1)
final_id = final_feature['user_id']
final_set = final_feature.iloc[:,used_feature]
predict = model.predict(final_set)
result = []
for i in range(len(predict)):
    if(predict[i] == 1):
        result.append(final_id.iloc[i])
print("最终提交数据：",len(result),"条")
result = pd.DataFrame(result)
result.to_csv('result.csv',index=None)

#
# #####


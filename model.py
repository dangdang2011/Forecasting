import pandas as pd
import numpy as np
import addFeature as af
from sklearn import preprocessing, model_selection
from sklearn.decomposition import PCA
import AddIntervalFeature as aif
import AddContinuousFeature as acf
from sklearn.model_selection import GridSearchCV


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
    #这里就是给列改名了
    activity_res.rename(columns={0: 'action_type_0',1: 'action_type_1',2: 'action_type_2',\
                                 3: 'action_type_3',4: 'action_type_4',5: 'action_type_5'  }, inplace=True)
    #print(activity_res.head())
    activity_page = temp_activity.groupby(['user_id','page'])['day_times'].size().unstack().fillna(0).reset_index()
    activity_page.rename(columns={0: 'action_page_0',1: 'action_page_1',2: 'action_page_2',\
                                 3: 'action_page_3',4: 'action_page_4'}, inplace=True)
    # print(activity_res.names())
    launch_res = temp_launch.groupby('user_id').count().reset_index()
    video_res = temp_video.groupby('user_id').count().reset_index()

    feature = pd.merge(launch_res,activity_res,on = 'user_id',how = 'left')
    feature = pd.merge(feature,activity_page,on='user_id',how='left')
    feature = pd.merge(feature,video_res,on = 'user_id',how='left')
    feature = pd.merge(temp_register,feature,on='user_id',how='left')
    feature = feature.fillna(0)

    #print("Feature",feature.info())

    feature = af.AddFeature(feature)

    launch_interval = aif.Add_launch_Interval_Feature(temp_launch)
    #这里增加了平均创作视频间隔
    create_interval=aif.Add_create_Interval_Feature(temp_video)
    #这里增加了连续登陆的最大天数
    launch_continuous=acf.Add_continuous_launch_Feature(temp_launch)

    feature=pd.merge(feature,create_interval,on='user_id',how='left').fillna(0)
    feature = pd.merge(feature, launch_interval, on='user_id', how='left')
    feature=pd.merge(feature,launch_continuous,on='user_id',how='left')

    feature = feature.fillna(0)
    #print("Feature",feature.info())
    #  创建一个PCA对象，并且将n_components的参数值设置为3。
    # 这里如果n_components的参数值为空将保留所有的特征。如果设置成‘mle’,那么会自动确定保留的特征数
    # copy:类型：bool，True或者False，缺省时默认为True。
    ## 意义：表示是否在运行算法时，将原始训练数据复制一份。若为True，则运行PCA算法后，原始训练数据的值不会有任何改变，
    # whiten:类型：bool，缺省时默认为False意义：白化。
    #used_feature = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    used_feature = []
    for i in range(4,31):
        used_feature.append(i)
    used_set = feature.iloc[:,used_feature]
    pca = PCA(n_components=4, copy=True, whiten=False)
    # 使用PCA对标准化后的特征进行降维
    used_set_pca = pca.fit_transform(used_set)
    # 查看降维后特征的维度，输出格式：(数据条目，特征数据)
    print('PCA降维后维度', used_set_pca.shape)
    # explained_variance_ratio_：array, [n_components]返回 所保留的n个成分各自的方差百分比,这里可以理解为单个变量方差贡献率，
    print('单个变量方差贡献率', pca.explained_variance_ratio_)
    PCA_feature = pd.DataFrame(used_set_pca)
    PCA_feature['user_id'] = feature['user_id']
    feature = pd.merge(feature, PCA_feature)
    #feature.to_csv('feature.csv')
    #print(feature[feature['user_id']==14808][['user_id','launch_continuous_count']])
    #print(temp_video[temp_video['user_id']==107685])
    #print("Feature",feature.info())
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

# add feature 部分有报错
#这里要引入addFeature模块，增加的特征是Max,Min,Max_type,Min_type,Median,Std,Skew,Kurt
#train_feature = af.AddFeature(train_feature)

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
print("1-23日活跃用户有：",active,"人")

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

# 我们先选择几种特征进行计算
#used_feature = [4,5,6,7,8,9,10,11]

# 将used_feature 作为待选特征，赋值给train_set，作为训练集输入模型
#train_set = train_feature.iloc[:,used_feature]
#used_feature = [4,5,6,7,8,9,10,11,12,13,14,15]
used_feature=[]
for i in range(4,35):
    used_feature.append(i)

train_set = train_feature.iloc[:,used_feature]
# print(train_set)

# !!!注意，这部分是用来验证模型好坏的，最终提交的部分里，模型暂时写死
#这个函数定义了拟合模型，可以选择不同的模型来拟合，用来快速测试
def modelUse(model_name,data_set,data_label,data_id):
    # 切分训练
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data_set, train_label, test_size=0.1,
                                                                        random_state=1017)
    model = model_name

    # 在这里我们选择model_Set中的一种model进行拟合,用data_set 和data_label来拟合
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
model_Set = [XGBClassifier(),GradientBoostingClassifier()]

for model in model_Set:
    print("在这里使用了模型：",model)
    modelUse(model_name=model,data_set=train_set,data_id=train_id,data_label=train_label)

## 模型调参
'''
def modelPara(data_train,data_label):
    param_xg_test = {
        # 'max_depth':[7,8,9,10],
        # 'min_child_weight':[5,6,7,8],
        # 'gamma': [i / 10.0 for i in range(2, 4)],
        'learning_rage':[i/10 for i in range(1,3)]
    }

    estimator = XGBClassifier(max_depth= 3, min_child_weight = 4,gamma=0.3)
    param_gbdt_test = {
    }
    # estimator = GradientBoostingClassifier(min_sam)

    gsearch = GridSearchCV(estimator, param_grid=param_xg_test, scoring='f1', cv=5)
    gsearch.fit(data_train, data_label)
    print("Parameter:",gsearch.best_params_,gsearch.best_score_)

modelPara(train_set,train_label)
'''
##

#####
#
# 建模，20180607测试（最下面的函数测试的）GBDT最优秀，准确率0.72左右，所以用gbdt提交
gbdt = GradientBoostingClassifier()
gbdt.fit(train_set,train_label)
# 提交,这里文件的名字都称为final
final_feature = trainInterval(1,30)
final_id = final_feature['user_id']
final_set = final_feature.iloc[:,used_feature]
result = []
predict = gbdt.predict(final_set)
print("最终预测了",len(predict),"条数据")
for i in range(len(predict)):
    if(predict[i] == 1):
        result.append(final_id.iloc[i])
print("其中，最终提交数据：",len(result),"条")
result = pd.DataFrame(result)
result.to_csv('result.csv',index=None)

#####

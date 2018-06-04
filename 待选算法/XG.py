from xgboost import XGBClassifier
import pandas as pd

file = pd.DataFrame(pd.read_table('bigdata.txt',
        names = ['user_id','app_launch','video_create','action_run','action_follow','action_repost','action_hate','label'],sep=' '))
# print(file.append('a'))
# print(file.iloc[1:3])
# print(file['label'])

Y_train = file['label']
X_train = file.iloc[:,[0,1,2,3,4,5,6]]

xg = XGBClassifier()
xg.fit(X_train,Y_train)
print(xg.predict(X_train))
print(Y_train)
from sklearn.linear_model import LogisticRegression
from numpy import *
import pandas as pd
file = pd.DataFrame(pd.read_table('bigdata.txt',
        names = ['user_id','app_launch','video_create','action_run','action_follow','action_repost','action_hate','label'],sep=' '))
Y_train = file['label']
X_train = file.iloc[:,[0,1,2,3,4,5,6]]

lr = LogisticRegression()
lr.fit(X_train,Y_train)

print(lr.predict(X_train))
print(Y_train)


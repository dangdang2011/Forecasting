from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd
file = pd.DataFrame(pd.read_csv('score.csv',
        names = ['user_id','app_launch','video_create','action_run','action_follow','action_repost','action_hate','label'],sep=' '))
Y_train = file['label']
X_train = file.iloc[:,[0,1,2,3,4,5,6]]

gbdt = GradientBoostingClassifier()
gbdt.fit(X_train,Y_train)
print(gbdt.predict(X_train))
print(Y_train)


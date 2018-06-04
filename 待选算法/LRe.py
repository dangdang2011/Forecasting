import pandas as pd
from sklearn import linear_model

import pandas as pd
file = pd.DataFrame(pd.read_table('bigdata.txt',
        names = ['user_id','app_launch','video_create','action_run','action_follow','action_repost','action_hate','label'],sep=' '))
Y_train = file['label']
X_train = file.iloc[:,[0,1,2,3,4,5,6]]

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
print("Coefficient",regr.coef_,"\n")
result = regr.predict(X_train)
for i in range(len(result)):
    print(result[i].round(2))
# print()
print(Y_train)


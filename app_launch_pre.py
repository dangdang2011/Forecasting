import pandas as pd
file = pd.DataFrame(pd.read_table('user_activity_log.txt',
        names = ['user_id','app_launch'],sep='\t'))
# Y_train = file['label']
# X_train = file.iloc[:,[0,1,2,3,4,5,6]]

print()


app_launch_res = file.groupby('user_id').count()
app_launch_res.to_csv('app_launch_res.csv')
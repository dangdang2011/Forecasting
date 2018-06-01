
import pandas as pd

file = pd.DataFrame(pd.read_table('user_activity_log.txt',
        names = ['user_id','day_times','page','video_id','author_id','action_type'],sep='\t'))

print(file['user_id'].groupby(file['action_type']).count()) # 输出每种action的总数量
print(file.groupby(['user_id','action_type'])[['day_times']].count().unstack())# count 操作查看每个user_id下action的数量。

grouped_file = file.groupby(['user_id','action_type'])[['day_times']].count().unstack().fillna(0)
print(grouped_file)

grouped_file = pd.DataFrame(grouped_file)
grouped_file.to_csv('action_type.csv')


# action_type
# 0    19798261
# 1      555671
# 2      206079
# 3       46078
# 4         157
# 5         982
import torch 
import pandas as pd 
import numpy as np
from utils import *
from datetime import datetime
import warnings
from datetime import timezone
import sklearn
from collections import Counter

#Multiclass imports
import tldextract
from urllib.parse import urlparse
from dateutil import parser

file_1 = 'Dataset/TweetsCOV19_p1_oct19-apr20.tsv'
file_2 = 'Dataset/TweetsCOV19_p2_may20.tsv'
file_3 = 'Dataset/TweetsCOV19_p3_jun20-dec20.tsv'

# using the smallest file to build pipeline for data pre-processing
df = pd.read_csv(file_2, sep='\t',names=['tweet_id',
                        'username',
                        'timestamp',
                        'followers',
                        'friends',
                        'retweets',
                        'favourites',
                        'entities',
                        'sentiments',
                        'mentions',
                        'hashtags',
                        'url'])

# Refer to this if you wanna merge the tsv files together. https://www.freecodecamp.org/news/how-to-combine-multiple-csv-files-with-8-lines-of-code-265183e0854/

df[['positive sentiment','negative sentiment']] = df.sentiments.str.split(" ",expand=True,)
y_value = df['retweets']
df_results = y_value.to_frame()
df = df.drop(columns=['sentiments','retweets']) # need to determine what fields are important and what can be removed 
df.insert(len(df.columns),'retweets',y_value) # shifted retweets to the back for easier viewing

## Numerical
print(f'starting df_num')
df_num = df[['followers','favourites','friends']]
for c in df_num.columns:
    df_num[f'{c}_z'] = z_transform(df_num[c].tolist())
    df_num[f'{c}_log'] = log_transform(df_num[c].tolist())
    df_num[f'{c}_cdf'] = cdf_transform(df_num[c].tolist())
    df_num[f'{c}_rank'] = rank_transform(df_num[c].tolist())

for c, d in zip(df_num.columns[:3], df_num.columns[1:3]):
    df_num[f'{c}_{d}'] = [a*b for a,b in zip(df_num[c].tolist(),df_num[d].tolist())]
    df_num[f'{c}_{d}_z'] = z_transform(df_num[f'{c}_{d}'])
    df_num[f'{c}_{d}_log'] = log_transform(df_num[f'{c}_{d}'])
    df_num[f'{c}_{d}_cdf'] = cdf_transform(df_num[f'{c}_{d}'])
    df_num[f'{c}_{d}_rank'] = rank_transform(df_num[f'{c}_{d}'])

df_num['followers_favourites_friends'] = [a*b for a,b in zip(df_num['followers_favourites'].tolist(),df_num['friends'].tolist())]
df_num['followers_favourites_friends_z'] = z_transform(df_num['followers_favourites_friends'])
df_num['followers_favourites_friends_log'] = log_transform(df_num['followers_favourites_friends'])
df_num['followers_favourites_friends_cdf'] = cdf_transform(df_num['followers_favourites_friends'])
df_num['followers_favourites_friends_rank'] = rank_transform(df_num['followers_favourites_friends'])

# normalizing
column_names = df_num.columns.tolist()
arr = df_num.to_numpy()
scaler = MinMaxScaler()
arr_norm = scaler.fit_transform(arr)
df_num = pd.DataFrame(data = arr_norm, columns = column_names)
print(f'finished df_num')

## Categorical
print(f'starting df_cat')
df_others = df[['tweet_id','username']]
df_categorical = df[['timestamp', 'positive sentiment', 'negative sentiment']]
df_onehot = df[['timestamp']]
indexing = index_users(df_others['username'].tolist())
user_counts = Counter(df_others['username'].tolist())
datetime_objects = [datetime.strptime(date, "%a %b %d %X %z %Y") for date in df_categorical['timestamp'].tolist()]
df_categorical['unix_timestamp'] = [dt.replace(tzinfo=timezone.utc).timestamp() for dt in datetime_objects]
df_categorical['user_count'] = [user_counts[i] for i in df_others['username'].tolist()]
df_others['user_id'] = [indexing[i] for i in df_others['username'].tolist()]
df_onehot['weekday'] = [date.weekday() for date in datetime_objects]
df_onehot['hour'] = [date.hour for date in datetime_objects]
# df_onehot['minute'] = [date.minute for date in datetime_objects]
df_onehot['day'] = [date.day for date in datetime_objects]
# df_onehot['week_of_month'] = [week_of_month(date) for date in datetime_objects]
df_onehot['month'] = [date.month for date in datetime_objects]
df_onehot['year'] = [date.year for date in datetime_objects]
df_categorical = df_categorical.drop(columns=['timestamp'])
df_others = df_others.drop(columns=['username'])
df_onehot = df_onehot.drop(columns=['timestamp'])

# one hot encode the timings
weekdays = pd.get_dummies(df_onehot.weekday, prefix='weekday') # df for weekdays
hours = pd.get_dummies(df_onehot.hour, prefix='hour') # df for hours
days = pd.get_dummies(df_onehot.day, prefix='day') # df for days
months = pd.get_dummies(df_onehot.month, prefix='month') # df for days
years = pd.get_dummies(df_onehot.year, prefix='year') # df for days
df_timings = pd.concat([weekdays, hours, days, months, years], axis=1)

#normalizing
column_names = df_categorical.columns.tolist()
arr = df_categorical.to_numpy()
scaler = MinMaxScaler()
arr_norm = scaler.fit_transform(arr)
df_categorical = pd.DataFrame(data = arr_norm, columns = column_names)
# df_categorical = pd.concat([df_others,df_categorical],axis=1)
df_categorical = pd.concat([df_others,df_categorical,df_timings],axis=1)
print(f'finished df_cat')



# read from the preprocessed file from clustering
user_metrics_features = pd.read_csv('processed_dataset/user_metrics_features.csv')
user_topics_features = pd.read_csv('processed_dataset/user_topics_features.csv')

user_metrics_onehot = pd.get_dummies(df['kmeans'], prefix="Cluster1")
user_topics_onehot = pd.get_dummies(df['kmeans'], prefix="Cluster2")

df_num_cat = pd.concat([df_categorical, df_num], axis=1)
# merge ur files with df_num_cat and save (we should get 301 dims)
df_results = pd.concat([df_num_cat, user_metrics_onehot, user_topics_onehot], axis=1)




# saving features and results 
#df_features.to_csv('processed_dataset/feature.csv', index=False)
df_results.to_csv('processed_dataset/result.csv', index=False)

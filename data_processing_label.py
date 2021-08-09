import torch 
import pandas as pd 
import numpy as np
from utils import *
from datetime import datetime
import warnings
from datetime import timezone

#Multiclass imports
import tldextract
from urllib.parse import urlparse
from dateutil import parser

file_1 = 'Dataset/TweetsCOV19_p1_oct19-apr20.tsv'
file_2 = 'Dataset/TweetsCOV19_p2_may20.tsv'
file_3 = 'Dataset/TweetsCOV19_p3_jun20-dec20.tsv'

# using the smallest file to build pipeline for data pre-processing
df = pd.read_csv(file_3, sep='\t',names=['tweet_id',
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

df[['positive sentiment','negative sentiment']] = df.sentiments.str.split(" ",expand=True,)
y_value = df['retweets']
df_results = y_value.to_frame()
df = df.drop(columns=['tweet_id','sentiments','retweets']) # need to determine what fields are important and what can be removed 
df.insert(len(df.columns),'retweets',y_value) # shifted retweets to the back for easier viewing

## Numerical
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

## Categorical
df_others = df[['tweet_id','username']]
df_categorical = df[['timestamp', 'positive sentiment', 'negative sentiment']]
indexing = index_users(df_others['username'].tolist())
datetime_objects = [datetime.strptime(date, "%a %b %d %X %z %Y") for date in df_categorical['timestamp'].tolist()]
df_categorical['unix_timestamp'] = [dt.replace(tzinfo=timezone.utc).timestamp() for dt in datetime_objects]
df_others['user_id'] = [indexing[i] for i in df_others['username'].tolist()]
df_categorical['weekday'] = [date.weekday() for date in datetime_objects]
df_categorical['hour'] = [date.hour for date in datetime_objects]
# df_categorical['minute'] = [date.minute for date in datetime_objects]
df_categorical['day'] = [date.day for date in datetime_objects]
# df_categorical['week_of_month'] = [week_of_month(date) for date in datetime_objects]
df_categorical['year'] = [date.year for date in datetime_objects]
df_categorical = df_categorical.drop(columns=['timestamp'])
df_others = df_others.drop(columns=['username'])

#normalizing
column_names = df_categorical.columns.tolist()
arr = df_categorical.to_numpy()
scaler = MinMaxScaler()
arr_norm = scaler.fit_transform(arr)
df_categorical = pd.DataFrame(data = arr_norm, columns = column_names)
df_categorical = pd.concat([df_others,df_categorical],axis=1)

## User Metrics (CLUSTER 1)
user_follow_mean_features = np.zeros(len(df))
user_follow_std_features = np.zeros(len(df))
user_friend_mean_features = np.zeros(len(df))
user_friend_std_features = np.zeros(len(df))
user_favorite_mean_features = np.zeros(len(df))
user_favorite_std_features = np.zeros(len(df))

user_entities_unique_features = np.zeros(len(df))
user_mentions_unique_features = np.zeros(len(df))
user_hashtags_unique_features = np.zeros(len(df))
user_urls_unique_features = np.zeros(len(df))

user_folos_change = np.zeros(len(df))
user_friends_change = np.zeros(len(df))
user_days_change = np.zeros(len(df))
user_folos_change = np.array(user_folos_change)
user_friends_change = np.array(user_friends_change)
user_days_change = np.array(user_days_change)

user_num = df['username'].nunique()
count = 0
for idx, (username, X_user) in tqdm(enumerate(df.groupby('username')), total=user_num):
    X_user = X_user.sort_values('timestamp', ascending=True)
    user_index = X_user.index.tolist()
    
    # User metric categories
    follow_tmp = []
    friend_tmp = []
    fav_tmp = []
    entities_tmp, mentions_tmp, hashtags_tmp, urls_tmp = [], [], [], []
    
    rows = []
    
    for fol, fri, fav, entities, mentions, hashtags, urls in X_user[['followers', 'friends', 'favourites', 'entities', 'mentions', 'hashtags', 'url']].values:
        follow_tmp.append(fol)
        friend_tmp.append(fri)
        fav_tmp.append(fav)
        entities_tmp.extend(entities)
        if type(mentions) == float:
            mentions_tmp.extend("null;")
        elif type(mentions) != float:
            mentions_tmp.extend(mentions)
        if type(hashtags) == float:
            hashtags_tmp.extend("null;")
        elif type(hashtags) != float:
            hashtags_tmp.extend(hashtags)
        if type(urls) == float:
            urls_tmp.extend("null;")
        elif type(urls) != float:
            urls_tmp.extend(urls)
    
    features = pd.DataFrame()
    # Followers
    follow_tmp = np.array(follow_tmp)
    features['user_follow_mean'] = [follow_tmp.mean()]
    features['user_follow_std'] = [follow_tmp.std()]
    # Friends
    friend_tmp = np.array(friend_tmp)
    features['user_friend_mean'] = [friend_tmp.mean()]
    features['user_friend_std'] = [friend_tmp.std()]
    # Favourites
    fav_tmp = np.array(fav_tmp)
    features['user_favourite_mean'] = [fav_tmp.mean()]
    features['user_favourite_std'] = [fav_tmp.std()]
    
    features['user_entities_unique'] = len(set(entities_tmp))
    features['user_mentions_unique'] = len(set(mentions_tmp))
    features['user_hashtags_unique'] = len(set(hashtags_tmp))
    features['user_urls_unique'] = len(set(urls_tmp))
    count += 1
    
    # Assign to whole user df
    user_follow_mean_features[user_index] = features['user_follow_mean']
    user_follow_std_features[user_index] = features['user_follow_std']
    user_friend_mean_features[user_index] = features['user_friend_mean']
    user_friend_std_features[user_index] = features['user_friend_std']
    user_favorite_mean_features[user_index] = features['user_favourite_mean']
    user_favorite_std_features[user_index] = features['user_favourite_std']
    print(features['user_favourite_std'])

    user_entities_unique_features[user_index] = features['user_entities_unique']
    user_mentions_unique_features[user_index] = features['user_mentions_unique']
    user_hashtags_unique_features[user_index] = features['user_hashtags_unique']
    user_urls_unique_features[user_index] = features['user_urls_unique']

# User Dynamics (CLUSTER 1)
df_user_metrics = pd.DataFrame()

df_user_metrics ['user_friend_mean'] = user_friend_mean_features
df_user_metrics ['user_friend_std'] = user_friend_std_features
df_user_metrics ['user_follow_mean'] = user_follow_mean_features
df_user_metrics ['user_follow_std'] = user_follow_std_features
df_user_metrics ['user_favorite_mean'] = user_favorite_mean_features
df_user_metrics ['user_favorite_std'] = user_favorite_std_features

df_user_metrics ['user_entities_unique'] = user_entities_unique_features
df_user_metrics ['user_mentions_unique'] = user_mentions_unique_features
df_user_metrics ['user_hashtags_unique'] = user_hashtags_unique_features
df_user_metrics ['user_urls_unique'] = user_urls_unique_features

df_changes = df[['username', 'timestamp', 'followers', 'friends']]
datetime_objects = [datetime.strptime(date, "%a %b %d %X %z %Y") for date in df_changes['timestamp'].tolist()]
df_changes['weekday'] = [date.weekday() for date in datetime_objects]
df_changes['hour'] = [date.hour for date in datetime_objects]
df_changes['day'] = [date.day for date in datetime_objects]
df_changes['week_of_month'] = [week_of_month(date) for date in datetime_objects]
user_num = df['username'].nunique()
timestamp = []

for idx, (username, X_user) in tqdm(enumerate(df_changes.groupby('username')), total=user_num):
    X_user = X_user.sort_values('timestamp', ascending=True)
    user_index = X_user.index.tolist()
    folos_temp = []
    friends_temp = []
    user_dates_temp = []
    user_folos_change_temp = [0]
    user_friends_change_temp = [0]
    user_days_change_temp = [0]
    user_months_change_temp = [0]
    temp_date = ''
    rows = ''
    for dt, folo, friend in X_user[['timestamp', 'followers', 'friends']].values:
        parsed_dt = parser.parse(dt)
        temp_date = parsed_dt
        folos_temp.append(folo)
        friends_temp.append(friend)
        user_dates_temp.append(parsed_dt)
            
        if len(folos_temp) > 1:
            user_folos_change_temp.append(folo - folos_temp[-2])
            user_friends_change_temp.append(friend - friends_temp[-2])
            user_days_change_temp.append((parsed_dt - user_dates_temp[-2]).days)

    increment_features = pd.DataFrame()
    increment_features['user_days_change_temp'] = user_days_change_temp
    increment_features['user_folos_change_temp'] = user_folos_change_temp
    increment_features['user_friends_change_temp'] = user_friends_change_temp
    user_days_change[user_index] = increment_features['user_days_change_temp']
    user_folos_change[user_index] = increment_features['user_folos_change_temp']
    user_friends_change[user_index] = increment_features['user_friends_change_temp']
    
df_user_change = pd.DataFrame()
df_user_change['user_days_change'] = user_days_change
df_user_change['user_folos_change'] = user_folos_change
df_user_change['user_friends_change'] = user_friends_change

df_metrics_dynamics = pd.merge(df_user_change, df_user_metrics)
user_metrics_features = apply_kmeans(df_metrics_dynamics, 100)

# User Topic Features (CLUSTER 2)
df_url = df[['url']]
df_url['url'].values
df_url_split = factorise_url(df_url)
df_multi_cat = df[['entities', 'mentions', 'hashtags']]
df_multi_cat_final = df_multi_cat.join(df_url_split)
df_preprocess = preprocess_entities(df_multi_cat_final)
df_preprocess = preprocess(df_preprocess, 'mentions')
df_preprocess = preprocess(df_preprocess, 'hashtags')
df_topic_features = extract_text(df_preprocess, 5)

user_topic_features= apply_kmeans(df_topic_features, 100)

## Merging your part using concat 
df_num_cat = pd.concat([df_categorical, df_num], axis=1)
df_topic_metrics = pd.concat([user_metrics_features, user_topic_features], axis=1)
df_features = pd.concat([df_num_cat, df_topic_metrics], axis=1)

# can test df_features shape should be 1912070x80

df_features.to_csv('processed_dataset/features.csv', index=False)
df_results.to_csv('processed_dataset/result.csv', index=False)






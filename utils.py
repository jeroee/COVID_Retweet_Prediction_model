import pandas as pd
import statistics
import math
from scipy.stats import norm
import matplotlib.pyplot as pltfre
import numpy as np
from urllib.parse import urlparse
import tldextract
import matplotlib.pyplot as plt

# numerical features
def z_transform(values): 
    '''
    z transformation (list)
    '''
    avg = statistics.mean(values)
    std = statistics.stdev(values)
    return [((value-avg)/std) for value in values]

def log_transform(values):
    '''
    log transformation (list)
    '''
    return [math.log(value+1) for value in values]

def cdf_transform(values):
    '''
    cdf transformation (list)
    '''
    return norm.cdf(values)

def rank_transform(values):
    '''
    rank transformation (list)
    '''
    return list(map({j: i for i, j in enumerate(sorted(set(values)))}.get, values))

def week_of_month(dt):
    '''
    getting the week of the month 0-4 (int) 
    '''
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(math.ceil(adjusted_dom/7.0))

# match each unique user to an index
def index_users(users):
    users_indexed = {k: v for v, k in enumerate(users)}
    return users_indexed

def factorise_url(df):
    url_list, scheme_list, domain_list, tld_list = [], [], [], []
    for i in df['url'].values:
        if i == 'null;' or type(i) == float:
            og_url = 'null'
            scheme = 'null'
            domain = 'null'
            tld = 'null'
        else: 
            og_url = i
            url_1 = tldextract.extract(i)
            url_2 = urlparse(i)
            
            scheme = url_2.scheme
            domain = url_1.domain
            tld = url_1.suffix
            
        url_list.append(og_url)
        scheme_list.append(scheme)
        domain_list.append(domain)
        tld_list.append(tld)
    
    df['url'] = url_list
    df['url_scheme'] = scheme_list
    df['url_domain'] = domain_list
    df['url_tld'] = tld_list
    return df

# preprocess entities
def preprocess_entities(df):
    values = [e.split(';')[:-1] for e in df['entities'].values]
    rows = []
    for x in values: 
        if x[0] == 'null':
            row = ['null']
        else:
            row = []
            for e in x:
                e = e.split(':')
                row.append(e[1])
        rows.append(row)
    df['entities'] = rows
    return df

# preprocess mentions, hashtags
def preprocess(df, colname):
    values = df[colname].values
    rows = []
    for x in values:
        if x == "null;" or type(x) == float:
            row = ['null']
        else:
            row = []
            for m in x.split():
                row.append(m)
        rows.append(row)
    df[colname] = rows
    return df

# SVD reduction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def tfidf_reduce(sentences, svd_dims):
    vectorizer = TfidfVectorizer()
    
    model = TruncatedSVD(n_components=svd_dims)
    vectors = vectorizer.fit_transform(sentences)
    col = model.fit_transform(vectors)
    col = pd.DataFrame(col)
    return col
#     dense = vectors.todense()
#     denselist = dense.tolist()
#     return denselist

# TFIDF embedding
from tqdm import tqdm

def extract_text(x_train, svd_dims):
    tweet_url_publisher = []
    for urls in tqdm(x_train['url'].values):
        publisher = ''
        if not urls[0] == 'null':
            for url in urls:
                if 'twitter.com' in url:
                    url = url.split('/')
                    if len(url) > 3:
                        publisher = url[3]
        tweet_url_publisher.append(publisher)
    
    x_train['tweet_url_publisher'] = tweet_url_publisher
    
    texts = []
    
    for m, e, h, tup in x_train[['mentions', 'entities', 'hashtags', 'tweet_url_publisher']].values:
        if len(m) == 0 or m[0] == 'null;':
            m = []
        if len(e) == 0 or e[0] == 'null;':
            e = []
        if len(h) == 0 or h[0] == 'null;':
            h = []
            
        # factorize entities
        e_div = []
        for ent in e:
            e_div.append(ent)
            ent_div = ent.split('_')
            if len(ent_div) > 1:
                e_div.extend(ent_div)
        text = m + e_div + h + [tup]
        
        texts.append(' '.join(text))

    features = tfidf_reduce(texts, svd_dims)
    features = features.add_prefix('TFIDF_SVD')
    
    return features

# Kmeans clustering
from sklearn.cluster import KMeans
def apply_kmeans(df, cluster_num=1000):
    kmeans = KMeans(n_clusters=cluster_num)
    kmeans.fit(df.iloc[:, 0:])
    df['kmeans'] = kmeans.labels_
    return df

# Normalizer
from sklearn.preprocessing import MinMaxScaler
def normalize(df):
#     val = np.log(val + 1)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(df_scaled)
    return normalized_df


def plot_curves(train_loss, val_loss, epochs):
    e = [i+1 for i in range(epochs)]
    plt.plot(e,train_loss, label='Training Loss')
    plt.plot(e,val_loss, label='Validation Loss')
    plt.xticks(np.arange(min(e), max(e)+1,1.0))
    plt.legend()
    plt.title('loss_graph', color='black')
    plt.xlabel('epoch', color='black')
    plt.ylabel('loss',color='black')
    plt.tick_params(colors='black')
    # plt.savefig(graph_name,dpi=100,bbox_inches = 'tight')
    plt.show()

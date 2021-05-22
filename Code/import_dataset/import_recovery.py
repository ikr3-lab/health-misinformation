import pandas as pd
import sys

absolute_path = sys.argv[1]
path_recovery = f"{absolute_path}/Data/data_raw/ReCOVery"

df_news = pd.read_csv(f"{path_recovery}/recovery-news-data.csv", parse_dates=['publish_date'])
df_tweet = pd.read_csv(f"{path_recovery}/recovery-social-media-data.csv")

dict_tweets = {}
for news in list(set(df_tweet['news_id'])):
    ls_tweet = list(df_tweet[df_tweet['news_id'] == news]['tweet_id'])
    dict_tweets[news] = ls_tweet

df_news['tweets'] = df_news['news_id'].apply(lambda x: dict_tweets[x] if x in dict_tweets.keys() else [None])

df_news = df_news[['news_id', 'publish_date', 'body_text', 'reliability', 'tweets']]

df_news.to_csv(f"{path_recovery}/ReCOVery_raw.csv", index=False)

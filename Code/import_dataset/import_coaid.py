import pandas as pd
import sys

path_github = "https://raw.githubusercontent.com/cuilimeng/CoAID/master/"
absolute_path = sys.argv[1]
path_data_out = f"{absolute_path}/Data/data_raw/CoAID"


def merge(folder, type_information, label, path=path_github):
    try:
        df = pd.read_csv(f"{path}{folder}{type_information}{label}COVID-19.csv", parse_dates=['publish_date'])
    except:
        df = pd.read_csv(f"{path}{folder}{type_information}{label}COVID-19.csv")
    df_tweet = pd.read_csv(f"{path}{folder}{type_information}{label}COVID-19_tweets.csv")
    df_reply = pd.read_csv(f"{path}{folder}{type_information}{label}COVID-19_tweets_replies.csv")
    df = df[['Unnamed: 0', 'type', 'content', 'publish_date']]
    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

    ls_tweet = []
    ls_reply = []
    ls_label = []
    for index, row in df.iterrows():
        if row['index'] in list(df_tweet['index']):
            ls_tweet.append(list(df_tweet[df_tweet['index'] == row['index']]['tweet_id']))
        else:
            ls_tweet.append([None])
        if row['index'] in list(df_reply['news_id']):
            ls_reply.append(list(df_reply[df_reply['news_id'] == row['index']]['reply_id']))
        else:
            ls_reply.append([None])
        if label == 'Fake':
            ls_label.append('Fake')
        else:
            ls_label.append('True')
    df['tweets'] = ls_tweet
    df['replies'] = ls_reply
    df['label'] = ls_label
    return df


def concatenation(type, label):
    df_1 = merge(folder='05-01-2020/', type_information=type, label=label)
    df_2 = merge(folder='07-01-2020/', type_information=type, label=label)
    df = pd.concat([df_1, df_2])
    return df


df_fake_news = concatenation(type='News', label='Fake')
df_true_news = concatenation(type='News', label='Real')

df_coaid = pd.concat([df_fake_news, df_true_news])
df_coaid.to_csv(f"{path_data_out}/CoAID_raw.csv", index=False)

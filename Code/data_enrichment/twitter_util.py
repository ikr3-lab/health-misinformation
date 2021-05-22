import tweepy
from tweepy import OAuthHandler
import json
import pandas as pd


def get_tweet_info(diz):
    df = pd.DataFrame(columns=['tweet', 'time', 'text', 'favorite_count', 'retweet_count', 'user_id'], index=[0])
    for tweet in diz.keys():
        diz_2 = {'tweet': [tweet]}
        for col in ['time', 'text', 'favorite_count', 'retweet_count', 'user_id']:
            if col in diz[tweet].keys():
                diz_2[col] = [diz[tweet][col]]
        dfx = pd.DataFrame.from_dict(diz_2, orient='columns')
        df = pd.concat([df, dfx])
    return df.iloc[1:, :]


def get_retweet_info(diz):
    df = pd.DataFrame(columns=['tweet', 'time', 'favorite_count', 'retweet_count', 'original_tweet', 'user_id'],
                      index=[0])
    for tweet in diz.keys():
        diz_2 = {'tweet': [tweet]}
        for col in ['time', 'favorite_count', 'retweet_count', 'original_tweet', 'user_id']:
            if col in diz[tweet].keys():
                diz_2[col] = [diz[tweet][col]]
        dfx = pd.DataFrame.from_dict(diz_2, orient='columns')
        df = pd.concat([df, dfx])
    return df.iloc[1:, :]


def get_reply_info(diz):
    df = pd.DataFrame(
        columns=['tweet', 'time', 'favorite_count', 'text', 'user_id', 'reply_user_screen', 'reply_user_id',
                 'user_screen'], index=[0])
    for tweet in diz.keys():
        diz_2 = {'tweet': [tweet]}
        for col in ['time', 'favorite_count', 'text', 'user_id', 'reply_user_screen', 'reply_user_id',
                    'user_screen']:
            if col in diz[tweet].keys():
                diz_2[col] = [diz[tweet][col]]
        dfx = pd.DataFrame.from_dict(diz_2, orient='columns')
        df = pd.concat([df, dfx])
    return df.iloc[1:, :]


def get_user_info(diz):
    df = pd.DataFrame(
        columns=['tweet', 'user_id', 'followers_count', 'user_description', 'user_favourite', 'user_verified',
                 'user_friend', 'default_profile_image'], index=[0])
    for tweet in diz.keys():
        if 'user_id' in diz[tweet].keys():
            diz_2 = {'tweet': [tweet]}
            for col in ['user_id', 'followers_count', 'user_description', 'user_favourite', 'user_verified',
                        'user_friend', 'default_profile_image']:
                if col in diz[tweet].keys():
                    diz_2[col] = [diz[tweet][col]]
            dfx = pd.DataFrame.from_dict(diz_2, orient='columns')
            df = pd.concat([df, dfx])
    return df.iloc[1:, :]


def create_api_twitter(path_secret):
    with open(path_secret, "r") as f:
        secret = json.load(f)
        consumer_key = secret["CONSUMER_KEY"]
        consumer_secret = secret["CONSUMER_SECRET"]

        access_token = secret["ACCESS_TOKEN"]
        access_token_secret = secret["ACCESS_TOKEN_SECRET"]

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    return api


class TwitterData:
    def __init__(self, df_name, path_secret):
        self.api = create_api_twitter(path_secret)
        self.df_name = df_name

    def take_id(self, id_tweet):
        tweet = self.api.get_status(id_tweet)
        return tweet

    def extract_tweet(self, ls_tweet):
        if len(ls_tweet) > 0 and pd.notna(ls_tweet[0]):
            diz = {}
            retweets_ls = []
            diz_rt = {}
            for tweet in ls_tweet:
                id_t = 'NO'
                try:
                    id_t = self.take_id(tweet)
                except:
                    continue
                if id_t != 'NO' and id_t.text[:2] != 'RT':
                    diz[tweet] = {}
                    try:
                        diz[tweet]['time'] = id_t.created_at
                    except:
                        continue
                    try:
                        diz[tweet]['text'] = id_t.text
                    except:
                        continue
                    try:
                        diz[tweet]['favorite_count'] = id_t.favorite_count
                    except:
                        continue
                    diz[tweet]['retweet_count'] = id_t.retweet_count
                    if self.df_name == 'CoAID' or self.df_name == 'ReCOVery':
                        if id_t.retweet_count > 0:
                            retweets_list = id_t.retweets()
                            for retweet in retweets_list:
                                diz_rt[retweet.id] = {}
                                diz_rt[retweet.id]['time'] = retweet.created_at
                                diz_rt[retweet.id]['favorite_count'] = retweet.favorite_count
                                diz_rt[retweet.id]['retweet_count'] = retweet.retweet_count
                                diz_rt[retweet.id]['original_tweet'] = retweet.retweeted_status.id_str
                                diz_rt[retweet.id]['user_id'] = retweet.user.id
                                diz_rt[retweet.id]['followers_count'] = retweet.user.followers_count
                                diz_rt[retweet.id]['user_description'] = retweet.user.description
                                diz_rt[retweet.id]['user_favourite'] = retweet.user.favourites_count
                                diz_rt[retweet.id]['user_verified'] = retweet.user.verified
                                diz_rt[retweet.id]['user_friend'] = retweet.user.friends_count
                                diz_rt[retweet.id]['default_profile_image'] = retweet.user.default_profile_image
                                retweets_ls.append(retweet.id)
                    try:
                        diz[tweet]['user_id'] = id_t.user.id
                    except:
                        continue
                    try:
                        diz[tweet]['followers_count'] = id_t.user.followers_count
                    except:
                        continue
                    try:
                        diz[tweet]['user_description'] = id_t.user.description
                    except:
                        continue
                    try:
                        diz[tweet]['user_favourite'] = id_t.user.favourites_count
                    except:
                        continue
                    try:
                        diz[tweet]['user_verified'] = id_t.user.verified
                    except:
                        continue
                    try:
                        diz[tweet]['user_friend'] = id_t.user.friends_count
                    except:
                        continue
                    try:
                        diz[tweet]['default_profile_image'] = id_t.user.default_profile_image
                    except:
                        continue
            if self.df_name == 'CoAID' or self.df_name == 'ReCOVery':
                if len(retweets_ls) == 0:
                    retweets_ls = [None]
                return [diz, retweets_ls, diz_rt]
            else:
                return diz
        else:
            if self.df_name == 'CoAID' or self.df_name == 'ReCOVery':
                return [None, [None], None]
            else:
                return None

    def extract_retweet(self, ls_tweet):
        if len(ls_tweet) > 0 and pd.notna(ls_tweet[0]):
            diz = {}
            for tweet in ls_tweet:
                id_t = 'NO'
                try:
                    id_t = self.take_id(tweet)
                    diz[tweet] = {}
                except:
                    continue
                if id_t != 'NO':
                    try:
                        diz[tweet]['time'] = id_t.created_at
                    except:
                        continue
                    try:
                        diz[tweet]['favorite_count'] = id_t.favorite_count
                    except:
                        continue
                    try:
                        diz[tweet]['retweet_count'] = id_t.retweet_count
                    except:
                        continue
                    try:
                        diz[tweet]['original_tweet'] = id_t.retweeted_status.id_str
                    except:
                        continue
                    try:
                        diz[tweet]['user_id'] = id_t.user.id
                    except:
                        continue
                    try:
                        diz[tweet]['followers_count'] = id_t.user.followers_count
                    except:
                        continue
                    try:
                        diz[tweet]['user_description'] = id_t.user.description
                    except:
                        continue
                    try:
                        diz[tweet]['user_favourite'] = id_t.user.favourites_count
                    except:
                        continue
                    try:
                        diz[tweet]['user_verified'] = id_t.user.verified
                    except:
                        continue
                    try:
                        diz[tweet]['user_friend'] = id_t.user.friends_count
                    except:
                        continue
                    try:
                        diz[tweet]['default_profile_image'] = id_t.user.default_profile_image
                    except:
                        continue
            return diz
        else:
            return None

    def extract_reply(self, ls_tweet):
        if len(ls_tweet) > 0 and pd.notna(ls_tweet[0]):
            diz = {}
            for tweet in ls_tweet:
                id_t = 'NO'
                try:
                    id_t = self.take_id(tweet)
                    diz[tweet] = {}
                except:
                    continue
                if id_t != 'NO':
                    try:
                        diz[tweet]['time'] = id_t.created_at
                    except:
                        continue
                    try:
                        diz[tweet]['favorite_count'] = id_t.favorite_count
                    except:
                        continue
                    try:
                        diz[tweet]['text'] = id_t.text
                    except:
                        continue
                    try:
                        diz[tweet]['user_screen'] = id_t.user.screen_name
                    except:
                        continue
                    try:
                        diz[tweet]['reply_user_screen'] = id_t.in_reply_to_screen_name
                    except:
                        continue
                    try:
                        diz[tweet]['reply_user_id'] = id_t.in_reply_to_user_id
                    except:
                        continue
                    try:
                        diz[tweet]['user_id'] = id_t.user.id
                    except:
                        continue
                    try:
                        diz[tweet]['followers_count'] = id_t.user.followers_count
                    except:
                        continue
                    try:
                        diz[tweet]['user_description'] = id_t.user.description
                    except:
                        continue
                    try:
                        diz[tweet]['user_favourite'] = id_t.user.favourites_count
                    except:
                        continue
                    try:
                        diz[tweet]['user_verified'] = id_t.user.verified
                    except:
                        continue
                    try:
                        diz[tweet]['user_friend'] = id_t.user.friends_count
                    except:
                        continue
                    try:
                        diz[tweet]['default_profile_image'] = id_t.user.default_profile_image
                    except:
                        continue
            return diz
        else:
            return None

    def download_twitter(self, df):
        if self.df_name == 'CoAID' or self.df_name == 'ReCOVery':
            df['var_temp'] = df['tweets'].apply(self.extract_tweet)
            df['tweet_info'] = df['var_temp'].apply(lambda x: x[0])
            df['retweets'] = df['var_temp'].apply(lambda x: x[1])
            df['retweets_count'] = df['retweets'].apply(lambda x: len(x) if x != [None] else 0)
            df['retweet_info'] = df['var_temp'].apply(lambda x: x[2])
            if self.df_name == 'CoAID':
                df['reply_info'] = df['replies'].apply(self.extract_reply)
        else:
            df['tweet_info'] = df['tweets'].apply(self.extract_tweet)
            df['retweet_info'] = df['retweets'].apply(self.extract_retweet)
            df['reply_info'] = df['replies'].apply(self.extract_reply)
        return df




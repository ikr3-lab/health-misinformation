import networkx as nx
import numpy as np
import pandas as pd
import spacy
from dateutil import parser
from networkx import dag_longest_path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from Code.utils.Processing import concat_twitter_data

absolute_path = "YOUR_ABSOLUTE_PATH"

path_resources = f"{absolute_path}/Resources"
NER_model = spacy.load(f"{path_resources}/en_core_sci_md-0.2.5/")


def macro(row, df_tweet, df_users_tweet, df_retweet, df_users_retweet):
    df_tweet_temp = df_tweet[df_tweet['tweet'].isin(row['tweets'])]
    df_rt_temp = df_retweet[df_retweet['original_tweet'].isin(row['tweets'])]
    df_tweet_temp['tweet'] = df_tweet_temp['tweet'].apply(lambda x: str(x))
    df_rt_temp['tweet'] = df_rt_temp['original_tweet'].apply(lambda x: str(x))

    number_tweet = df_tweet_temp.shape[0]
    number_retweet = df_rt_temp.shape[0]
    if number_tweet > 0 and number_retweet > 0:
        df_merge = pd.merge(df_tweet_temp, df_rt_temp, left_on='tweet', right_on='original_tweet',
                            suffixes=('_tweet', '_retweet'))
    else:
        df_merge = pd.DataFrame()

    df_utenti_temp = df_users_tweet[df_users_tweet['tweet'].isin(row['tweets'])]
    df_utenti_temp = df_utenti_temp.drop_duplicates('user_id')
    df_utenti_rt_temp = df_users_retweet[df_users_retweet['user_id'].isin(df_rt_temp['user_id'])]
    df_utenti_rt_temp = df_utenti_rt_temp.drop_duplicates('user_id')

    # ---------VARIABILI STRUTTURALI---------------------

    if number_retweet > 0:
        number_tweet_w_retweet = df_tweet_temp[df_tweet_temp['retweet_count'] > 0].shape[0]
    else:
        number_tweet_w_retweet = 0

    if number_tweet == 0:
        s1 = 0
    elif number_tweet > 0 and number_retweet == 0:
        s1 = 1
    else:
        s1 = 2

    if number_tweet > 0:
        s2 = number_tweet + number_retweet
    else:
        s2 = 0

    if number_tweet > 0 and number_retweet > 0:
        s3 = (max(df_tweet_temp['retweet_count'].max(), df_rt_temp['retweet_count'].max()))
    else:
        s3 = 0

    if number_tweet > 0:
        s4 = number_tweet
    else:
        s4 = 0

    if number_tweet == 0:
        s5 = 0
    elif number_tweet > 0 and number_retweet == 0:
        s5 = 1
    else:
        s5 = 2

    s6 = number_tweet_w_retweet

    if number_tweet > 0:
        s7 = number_tweet_w_retweet / number_tweet
    else:
        s7 = 0

    # ---------VARIABILI ENGAGEMENT---------------------

    if number_tweet > 0 and number_retweet > 0:
        e1 = (df_tweet_temp['favorite_count'].sum() + df_rt_temp['favorite_count'].sum()) / (
                number_tweet + number_retweet)  # Favorite medio nella rete macro:
    else:
        e1 = 0

    if number_tweet > 0 and number_retweet > 0:
        e2 = max(df_tweet_temp['favorite_count'].max(),
                 df_rt_temp['favorite_count'].max())  # Favorite max nella rete macro:
    else:
        e2 = 0

    if number_tweet > 0 and number_retweet > 0:
        e3 = df_tweet_temp[df_tweet_temp['retweet_count'] > 0][
            'favorite_count'].mean()  # Favorite medio nella cascata:
    else:
        e3 = 0

    # ---------VARIABILI UTENTI---------------------

    if df_utenti_temp.shape[0] > 0:
        u1 = df_utenti_temp['followers_count'].median()  # Median follower
    else:
        u1 = 0

    if df_utenti_temp.shape[0] > 0:
        u2 = df_utenti_temp['followers_count'].max()  # Max follower
    else:
        u2 = 0

    if df_utenti_rt_temp.shape[0] > 0:
        u3 = df_utenti_rt_temp['followers_count'].median()  # Median follower rt
    else:
        u3 = 0

    if df_utenti_temp.shape[0] > 0:
        u5 = df_utenti_temp['user_friend'].median()  # Median followee
    else:
        u5 = 0

    if df_utenti_temp.shape[0] > 0:
        u6 = df_utenti_temp['user_friend'].max()  # Max followee
    else:
        u6 = 0

    if df_utenti_rt_temp.shape[0] > 0:
        u7 = df_utenti_rt_temp['user_friend'].median()  # Median followee rt
    else:
        u7 = 0

    if df_utenti_temp.shape[0] > 0:
        df_utenti_temp['temp'] = df_utenti_temp['user_description'].apply(NER)
        df_utenti_temp[['termini_biomedici', 'termini_biomedici_unici']] = pd.DataFrame(
            df_utenti_temp.temp.tolist(),
            index=df_utenti_temp.index)
        u9 = df_utenti_temp['termini_biomedici'].mean()  # Termini medici nel 1 lv
        if df_utenti_rt_temp.shape[0] > 0:
            df_utenti_rt_temp['temp'] = df_utenti_rt_temp['user_description'].apply(NER)
            df_utenti_rt_temp[['termini_biomedici', 'termini_biomedici_unici']] = pd.DataFrame(
                df_utenti_rt_temp.temp.tolist(), index=df_utenti_rt_temp.index)
            u10 = df_utenti_rt_temp['termini_biomedici'].mean()
        else:
            u10 = 0
    else:
        u9 = 0
        u10 = 0

    if number_tweet > 0:
        cu1 = len(list(set(df_tweet_temp['user_id']))) / number_tweet
    else:
        cu1 = 0

    if number_tweet > 0:
        cu2 = df_tweet_temp.groupby('user_id').size().max()
    else:
        cu2 = 0

    if number_retweet > 0:
        cu3 = len(list(set(df_rt_temp['user_id']))) / number_retweet
    else:
        cu3 = 0

    if number_retweet > 0:
        cu4 = df_rt_temp.groupby('user_id').size().max()
    else:
        cu4 = 0

    # -----------------VARIABILI TEMPORALI-------------
    df_tweet_temp.reset_index(drop=True)
    if df_merge.shape[0] > 0:
        t1 = np.mean([x.total_seconds() for x in df_merge['time_retweet'] - df_merge['time_tweet']])
    else:
        t1 = -1

    try:
        t2 = (max(df_rt_temp['time']) - min(df_tweet_temp['time'])).total_seconds()
    except:
        t2 = -1

    if number_tweet > 0:
        t3 = (df_tweet_temp.loc[df_tweet_temp['retweet_count'].idxmax(), 'time'] - min(
            df_tweet_temp['time'])).total_seconds()
    else:
        t3 = -1

    if number_tweet > 0:
        t4 = (max(df_tweet_temp['time']) - min(df_tweet_temp['time'])).total_seconds()
    else:
        t4 = -1

    if number_tweet > 0:
        hub_rt = df_tweet_temp.loc[df_tweet_temp['retweet_count'].idxmax(), 'tweet']
    else:
        hub_rt = 0

    try:
        t5 = [x.total_seconds() for x in (max(df_merge[df_merge['original_tweet'] == hub_rt]['time_retweet']) -
                                          df_tweet_temp[df_tweet_temp['tweet'] == hub_rt]['time'])][0]
    except:
        t5 = -1

    if number_retweet > 0 and hub_rt != 0 and df_merge.shape[0] > 0:
        ls_t6_temp = list(df_merge[df_merge['original_tweet'] == hub_rt]['time_retweet'])
        t6 = np.mean(
            [(x - list(df_tweet_temp[df_tweet_temp['tweet'] == hub_rt]['time'])[0]).total_seconds() for x in
             ls_t6_temp])
    else:
        t6 = -1

    publish_date = 'no'
    try:
        publish_date = parser.parse(str(row['publish_date'])).replace(tzinfo=None)
    except:
        publish_date = 'no'

    if number_tweet > 0 and publish_date != 'no':
        ls_time = list(df_tweet_temp['time'])
        ls_x = [(x - publish_date).total_seconds() for x in ls_time]
        t7 = np.mean(ls_x)
    else:
        t7 = -1

# -------VAR NEL DF-----------

    return {'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4, 'S5': s5, 'S6': s6, 'S7': s7, 'E1': e1,
            'E2': e2, 'E3': e3, 'U1': u1, 'U2': u2, 'U3': u3, 'U5': u5, 'U6': u6, 'U7': u7,
            'U9': u9, 'U10': u10, 'CU1': cu1, 'CU2': cu2, 'CU3': cu3, 'CU4': cu4, 'T1': t1,
            'T2': t2, 'T3': t3, 'T4': t4, 'T5': t5, 'T6': t6, 'T7': t7}


def NER(text):
    if pd.notna(text):
        doc = NER_model(text)
        lungh = len(text)
        list_biomedical = [str(x).lower() for x in list(doc.ents)]
        num_ents = len(list_biomedical)
        num_unique_ents = len(list(set(list_biomedical)))
        return [num_ents / lungh, num_unique_ents / lungh]
    else:
        return [None, None]


# Sentiment
def sentiment_score(sentence):
    analyser = SentimentIntensityAnalyzer()
    sentence = str(sentence)
    if len(sentence) == 0:
        return None
    score = analyser.polarity_scores(sentence)
    return score['compound']


def sentiment_label(sentence):
    analyser = SentimentIntensityAnalyzer()
    sentence = str(sentence)
    if len(sentence) == 0:
        return None
    score = analyser.polarity_scores(sentence)
    if score['compound'] >= 0.05:
        return 'Pos'
    elif score['compound'] <= -0.05:
        return 'Neg'
    else:
        return 'Neu'


def aggregation_twitter_data(path_data_twitter, df_name):
    df_tweet = concat_twitter_data(f"{path_data_twitter}/{df_name}/Tweet/")
    df_tweet.to_csv(f"{path_data_twitter}/{df_name}_tweet.csv", index=False)
    df_users_tweet = concat_twitter_data(f"{path_data_twitter}/{df_name}/Users_tweet/")
    df_users_tweet.to_csv(f"{path_data_twitter}/{df_name}_df_users_tweet.csv", index=False)

    df_retweet = concat_twitter_data(f"{path_data_twitter}/{df_name}/Retweet/")
    df_retweet.to_csv(f"{path_data_twitter}/{df_name}_retweet.csv", index=False)
    df_users_retweet = concat_twitter_data(f"{path_data_twitter}/{df_name}/Users_retweet/")
    df_users_retweet.to_csv(f"{path_data_twitter}/{df_name}_df_users_retweet.csv", index=False)

    if df_name != 'ReCOVery':
        df_reply = concat_twitter_data(f"{path_data_twitter}/{df_name}/Reply/")
        df_reply.to_csv(f"{path_data_twitter}/{df_name}_reply.csv", index=False)
        df_users_reply = concat_twitter_data(f"{path_data_twitter}/{df_name}/Users_reply/")
        df_users_reply.to_csv(f"{path_data_twitter}/{df_name}_df_users_reply.csv", index=False)

        return df_tweet, df_users_tweet, df_retweet, df_users_retweet, df_reply, df_users_reply
    return df_tweet, df_users_tweet, df_retweet, df_users_retweet


def build_graph(ls_nodes, ls_edges):
    G = nx.DiGraph()
    G.add_nodes_from(ls_nodes)
    G.add_edges_from(ls_edges)
    return G


def micro(row, df_tweet, df_reply, df_utenti_reply):
    if row['replies'][0] != 'None':
        df_reply_temp = df_reply[df_reply['tweet'].isin(row['replies'])]
        df_tweet_temp = df_tweet[df_tweet['tweet'].isin(row['tweets'])]
        df_utenti_reply_temp = df_utenti_reply[df_utenti_reply['tweet'].isin(row['replies'])]

        num_reply = df_reply_temp.shape[0]

        # -----GRAFO-----
        ls_nodes = df_tweet_temp['user_id']
        ls_nodes.append(df_reply_temp['user_id'])
        ls_nodes.append(df_reply_temp['reply_user_id'])
        ls_edges = [(x, y) for x, y in zip(df_reply_temp['user_id'], df_reply_temp['reply_user_id'])]
        ls_edges = [x for x in ls_edges if x[0] != x[1]]
        G = build_graph(ls_nodes, ls_edges)

        # ----Strutturali-----

        try:
            ls_s10 = len(dag_longest_path(G))
        except:
            ls_s10 = 0

        if len(G) > 0:
            ls_s11 = len([g for n, g in G.out_degree() if g > 0])
            ls_s12 = max([g for n, g in G.in_degree()])
        else:
            ls_s11 = 0
            ls_s12 = 0

        if num_reply > 0:
            ls_s13 = df_reply_temp.drop_duplicates('reply_user_id').shape[0]
        else:
            ls_s13 = 0

        if len(G) > 0:
            ls_s14 = df_reply_temp.drop_duplicates('reply_user_id').shape[0] / len(G)
        else:
            ls_s14 = 0

        if num_reply > 0:
            ls_e4 = df_reply_temp['favorite_count'].mean()
            ls_e5 = df_reply_temp['favorite_count'].max()
        else:
            ls_e4 = 0
            ls_e5 = 0

        if num_reply > 0:
            ls_u4 = df_utenti_reply_temp.drop_duplicates(subset='user_id')['followers_count'].median()
            ls_u8 = df_utenti_reply_temp.drop_duplicates(subset='user_id')['user_friend'].median()
        else:
            ls_u4 = 0
            ls_u8 = 0

        if df_utenti_reply_temp.shape[0] > 0:
            ls_cu5 = num_reply / len(set(df_utenti_reply_temp['user_id']))
        else:
            ls_cu5 = 0

        if num_reply > 0:
            ls_cu6 = df_reply_temp.groupby('user_id').size().max()
        else:
            ls_cu6 = 0

        # ------Temporali-----
        if df_tweet_temp.shape[0] > 0 and num_reply > 0:
            df_reply_tweet = pd.merge(df_tweet_temp, df_reply_temp, left_on='user_id', right_on='reply_user_id',
                                      suffixes=('_tweet', '_reply'))
            num_reply_w_tweet = df_reply_tweet.shape[0]
        else:
            df_reply_tweet = pd.DataFrame()
            num_reply_w_tweet = 0

        if num_reply > 0:
            df_reply_reply = pd.concat(
                [df_reply_temp[df_reply_temp['user_id'].isin(df_reply_temp['reply_user_id'])],
                 df_reply_temp[df_reply_temp['reply_user_id'].isin(df_reply_temp['user_id'])]])
            df_reply_reply.drop_duplicates(inplace=True)
        else:
            df_reply_reply = pd.DataFrame()

        if num_reply_w_tweet > 0 and df_reply_reply.shape[0] > 0:
            df_replies_to_replies = pd.merge(df_reply_reply,
                                             df_reply_tweet[['user_id_reply', 'reply_user_id', 'time_reply']],
                                             left_on='reply_user_id', right_on='user_id_reply')
        else:
            df_replies_to_replies = pd.DataFrame()

        if num_reply_w_tweet > 0:
            t9_a = (df_reply_tweet['time_reply'] - df_reply_tweet['time_tweet'])
            t9_a = [x.total_seconds() for x in t9_a]
            ls_t10 = np.mean(t9_a)
            if df_replies_to_replies.shape[0] > 0 and len(t9_a) > 0:
                t9_b = (df_replies_to_replies['time'] - df_replies_to_replies['time_reply'])
                t9_b = [x.total_seconds() for x in t9_b]
                ls_t9 = np.mean(t9_a + t9_b)
            else:
                ls_t9 = -1
        else:
            ls_t10 = -1
            ls_t9 = -1

        if num_reply_w_tweet > 0:
            ls_t11 = (max(df_reply_tweet['time_reply']) - min(df_reply_tweet['time_tweet'])).total_seconds()
        else:
            ls_t11 = -1

        # -----LINGUISTICHE----
        ls_pol = list(df_reply_temp['text'].apply(sentiment_label))
        ls_pol_score = list(df_reply_temp['text'].apply(sentiment_score))
        if len(ls_pol) > 0 and 'Neg' in ls_pol:
            ls_l1 = len([x for x in ls_pol if x == 'Pos']) / len([x for x in ls_pol if x == 'Neg'])
            ls_l2 = np.mean(ls_pol_score)
        else:
            ls_l1 = 999
            ls_l2 = 999

        if num_reply_w_tweet > 0:
            ls_l3 = np.mean(df_reply_tweet['text_reply'].apply(sentiment_score))
        else:
            ls_l3 = 999

        if num_reply_w_tweet > 0:
            hub = df_reply_tweet.groupby('reply_user_id').size().idxmax()
            ls_l4 = np.mean(
                df_reply_tweet[df_reply_tweet['reply_user_id'] == hub]['text_reply'].apply(sentiment_score))
        else:
            ls_l4 = 999

    else:
        ls_s10 = 0
        ls_s11 = 0
        ls_s12 = 0
        ls_s13 = 0
        ls_s14 = 0
        ls_t9 = -1
        ls_t10 = -1
        ls_t11 = -1
        ls_l1 = 999
        ls_l2 = 999
        ls_l3 = 999
        ls_l4 = 999
        ls_u4 = 0
        ls_u8 = 0
        ls_cu5 = 0
        ls_cu6 = 0
        ls_e4 = 0
        ls_e5 = 0

    return {'S10': ls_s10, 'S11': ls_s11, 'S12': ls_s12, 'S13': ls_s13, 'S14': ls_s14, 'T9': ls_t9, 'T10': ls_t10,
            'T11': ls_t11, 'L1': ls_l1, 'L2': ls_l2, 'L3': ls_l3, 'L4': ls_l4, 'U4': ls_u4, 'U8': ls_u8, 'E4': ls_e4,
            'E5': ls_e5, 'CU5': ls_cu5, 'CU6': ls_cu6}

def parse_date_tweet(x):
  # return parser.parse(x)
  try:
    return parser.parse(str(x))
  except:
    return x
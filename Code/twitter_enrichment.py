from __future__ import absolute_import, print_function
import pandas as pd

from Code.data_enrichment import twitter_util
from Code.utils.Processing import str_to_list

ls_df = ['CoAID', 'FakeHealth_Release', 'ReCOVery', 'FakeHealth_Story']

absolute_path = "YOUR_ABSOLUTE_PATH"
path_data = f"{absolute_path}/Data/data_cleaned"
path_data_twitter = f"{absolute_path}/Data/data_twitter"
path_secret = f"{absolute_path}/Code/data_enrichment/secret.json"

for df_name in ls_df:

    df = pd.read_csv(f"{path_data}/{df_name}.csv", parse_dates=['publish_date'])

    # Parsing network data
    df['tweets'] = df['tweets'].apply(str_to_list)
    if 'FakeHealth' in df_name:
        df['retweets'] = df['retweets'].apply(str_to_list)
    if df_name != 'ReCOVery':
        df['replies'] = df['replies'].apply(str_to_list)

    # With this block is possible to download the Twitter information for every medical information separately and
    # store 1 csv for every kind of twitter data for each medical information
    min = 0
    max = df.shape[0]
    step = 1
    dfx = df
    dfx = dfx.reset_index()
    indexes = list(range(min, max, step))

    twitterdata = twitter_util.TwitterData(df_name=df_name, path_secret=path_secret)
    for ind in range(len(indexes)):

        print(indexes[ind])

        if ind < len(indexes) - 1:
            df = dfx.iloc[indexes[ind]:indexes[ind + 1]]
        else:
            df = dfx.iloc[indexes[ind]:max]

        df = twitterdata.download_twitter(df)

        if df['tweets'][indexes[ind]][0] is not None:
            df_users_tweet = twitter_util.get_user_info(df['tweet_info'][indexes[ind]])
            df_users_tweet.to_csv(f"{path_data_twitter}/{df_name}/Users_tweet/df_users_tweet_{str(indexes[ind])}.csv",
                                  index=False)
            df_tweet = twitter_util.get_tweet_info(df['tweet_info'][indexes[ind]])
            df_tweet.to_csv(f"{path_data_twitter}/{df_name}/Tweet/df_tweet_{str(indexes[ind])}.csv", index=False)

        if df['retweets'][indexes[ind]][0] is not None:
            df_users_retweet = twitter_util.get_user_info(df['retweet_info'][indexes[ind]])
            df_users_retweet.to_csv(
                f"{path_data_twitter}/{df_name}/Users_retweet/df_users_retweet_{str(indexes[ind])}.csv", index=False)
            df_retweet = twitter_util.get_retweet_info(df['retweet_info'][indexes[ind]])
            df_retweet.to_csv(f"{path_data_twitter}/{df_name}/Retweet/df_retweet_{str(indexes[ind])}.csv", index=False)

        if df['replies'][indexes[ind]][0] is not None:
            df_users_reply = twitter_util.get_user_info(df['reply_info'][indexes[ind]])
            df_users_reply.to_csv(f"{path_data_twitter}/{df_name}/Users_reply/df_users_reply_{str(indexes[ind])}.csv",
                                  index=False)
            df_reply = twitter_util.get_reply_info(df['reply_info'][indexes[ind]])
            df_reply.to_csv(f"{path_data_twitter}/{df_name}/Reply/df_reply_{str(indexes[ind])}.csv", index=False)

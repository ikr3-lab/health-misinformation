import pandas as pd
import json
import requests


def extract_liwc_features(absolute_path):
    url = 'https://api.receptiviti.com/v1/score'
    api_key = 'your_api_key'
    api_secret = 'your_api_secret'

    path_data = f"{absolute_path}/Data/data_w_feature"
    path_data_out = f"{absolute_path}/Data/data_liwc"

    ls_df = ['CoAID', 'FakeHealth_Release', 'ReCOVery', 'FakeHealth_Story']

    for df_name in ls_df:

        print(df_name)

        df = pd.read_csv(f"{path_data}/{df_name}.csv", parse_dates=['publish_date'])
        df_temp = pd.DataFrame()

        for ind, row in df.iterrows():
            data = json.dumps({'content': row['text']})
            resp = requests.post(url, auth=(api_key, api_secret), data=data)
            x = resp.json()['results']
            diz = x[0]['liwc']['scores']['categories']
            diz['news_id'] = row['news_id']
            dfx = pd.DataFrame.transpose(pd.DataFrame.from_dict(diz, orient='index'))
            df_temp = pd.concat([df_temp, dfx])

        df_fin = pd.merge(df[['news_id', 'class']], df_temp, on='news_id')
        df_fin.to_csv(f"{path_data_out}/{df_name}.csv", index=False)

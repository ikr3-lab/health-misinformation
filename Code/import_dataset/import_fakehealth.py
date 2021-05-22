import pandas as pd
import requests
import json
import re
from os import listdir
from os.path import isfile, join
import sys

absolute_path = sys.argv[1]
path_fakehealth = f"{absolute_path}/Data/data_raw/FakeHealth"

ls_path_release = [join(f"{path_fakehealth}/HealthRelease", f) for f in listdir(f"{path_fakehealth}/HealthRelease") if
                   isfile(join(f"{path_fakehealth}/HealthRelease", f))]
ls_path_story = [join(f"{path_fakehealth}/HealthStory", f) for f in listdir(f"{path_fakehealth}/HealthStory") if
                 isfile(join(f"{path_fakehealth}/HealthStory", f))]


def twitter_creator(metadata, key):
    ls_keys = list(metadata.keys())
    if 'twitter' in ls_keys:
        if key in metadata['twitter'].keys():
            return metadata['twitter'][key]
        else:
            return None
    else:
        return None


def download(ls_path):
    diz = {}
    ls_id = []
    if 'news' in ls_path[0]:
        nome_id = 'news_reviews_'
    else:
        nome_id = 'story_reviews_'
    for file in ls_path:
        f = open(file)
        try:
            content = json.loads(f)
        except:
            content = json.load(f)
        ls_id.append(nome_id + re.findall(r'\d+', file)[0])
        diz[nome_id + re.findall(r'\d+', file)[0]] = {}
        for field in content:
            diz[nome_id + re.findall(r'\d+', file)[0]].update({field: content[field]})
    df = pd.DataFrame.from_dict(diz, orient='index').reset_index()
    df['twitter_creator'] = df['meta_data'].apply(twitter_creator, key='creator')
    df['keywords'] = df['meta_data'].apply(lambda x: x['keywords'] if 'keywords' in list(x.keys()) else None)
    return df


df_release = download(ls_path_release)
df_story = download(ls_path_story)

df_txt = pd.concat([df_release, df_story])
df_txt['type'] = df_txt['index'].apply(lambda x: 'story' if 'story' in x else 'release')


def ground_truth(path):
    ls_id = []
    ls_rating = []
    ls_category = []
    ls_criteria = []
    j = requests.get(path)
    content = json.loads(j.content)
    for i in range(len(content)):
        ls_id.append(content[i]['news_id'])
        ls_rating.append(content[i]['rating'])
        ls_category.append(content[i]['category'])
        ls_temp = []
        for j in range(len(content[i]['criteria'])):
            ls_temp.append(content[i]['criteria'][j]['answer'])
        ls_criteria.append(ls_temp)
    rating = pd.DataFrame({'index': ls_id, 'category': ls_category, 'rating': ls_rating, 'criteria': ls_criteria})
    criteria = pd.DataFrame(rating["criteria"].to_list(),
                            columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10'])
    df = pd.merge(rating, criteria, left_index=True, right_index=True)
    return df


release = ground_truth(
    'https://raw.githubusercontent.com/EnyanDai/FakeHealth/master/dataset/reviews/HealthRelease.json')
story = ground_truth('https://raw.githubusercontent.com/EnyanDai/FakeHealth/master/dataset/reviews/HealthStory.json')

df_ground_truth = pd.concat([release, story])


def engagement(path):
    diz = {}
    j = requests.get(path)
    content = json.loads(j.content)
    if 'Story' in path:
        ls_id = list(df_txt[df_txt['type'] == 'story']['index'])
    else:
        ls_id = list(df_txt[df_txt['type'] == 'release']['index'])
    for i in ls_id:
        try:
            diz[i] = {'tweets': content[i]['tweets'], 'replies': content[i]['replies'],
                      'retweets': content[i]['retweets']}
        except:
            print(i)
    df = pd.DataFrame.from_dict(diz, orient='index').reset_index()
    df.columns = ['index'] + list(df.columns[1:])
    return df


engagement_story = engagement(
    'https://raw.githubusercontent.com/EnyanDai/FakeHealth/master/dataset/engagements/HealthStory.json')
engagement_release = engagement(
    'https://raw.githubusercontent.com/EnyanDai/FakeHealth/master/dataset/engagements/HealthRelease.json')

df_engagement = pd.concat([engagement_release, engagement_story])

df = pd.merge(df_txt, df_ground_truth, how='inner', on='index')
df = pd.merge(df, df_engagement, how='left', on='index')

df.to_csv(f"{path_fakehealth}/FakeHealth_raw.csv", index=False)

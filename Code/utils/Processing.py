import os
import pandas as pd
import numpy as np
from dateutil import parser
import datetime


def split_fakehealth(df):
    df['split'] = df['news_id'].apply(lambda x: x.split('_')[0])
    df_release = df[df['split'] == 'news']
    df_story = df[df['split'] == 'story']
    return df_release.drop(columns='split'), df_story.drop(columns='split')


def date_conversion_coaid_recovery(date):
    try:
        return parser.parse(date)
    except:
        return np.nan


def date_conversion_fakehealth(date):
    if pd.isnull(date):
        return np.nan
    else:
        try:
            date = int(date)
        except:
            date = int(date.split('.')[0])
        x = datetime.datetime.fromtimestamp(int(date)).strftime("%Y-%m-%d %H:%M:%S")
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def ground_truth_fakehealth(criteria):
    criteria_int = [1 if c == "Satisfactory" else 0 for c in str_to_list(criteria)]
    points = sum(criteria_int) / 2
    if points < 3:
        return 0
    else:
        return 1


def select_type_coaid(df):
    return df[df['type'] == 'article']


def uniform_variables_name(df_name):
    name = df_name.split('_')[0]
    if name == 'FakeHealth':
        dict = {'index': 'news_id', 'publish_date': 'publish_date', 'text': 'text', 'tweets': 'tweets',
                'retweets': 'retweets', 'replies': 'replies', 'class': 'class'}
    elif name == 'CoAID':
        dict = {'index': 'news_id', 'publish_date': 'publish_date', 'content': 'text', 'tweets': 'tweets',
                'replies': 'replies', 'label': 'class'}
    else:
        dict = {'news_id': 'news_id', 'publish_date': 'publish_date', 'body_text': 'text', 'tweets': 'tweets',
                'reliability': 'class'}
    return dict, dict.keys()


def del_col_0(df):
    gt = list((df.describe().loc['max', :] == 0) & (df.describe().loc['min', :] == 0))
    col = list(df.columns)
    ls_col_ok = []
    for coppia in zip(gt, col):
        if not coppia[0]:
            ls_col_ok.append(coppia[1])
    return ls_col_ok


def tipo_var(variabile):

    y_column = 'class'

    X_appoggio = ['news_id', 'publish_date', 'text', 'tweets', 'retweets', 'replies', 'len', 'emoction', 'len_text']

    X_token = ['token', 'token_clean', 'token_lem', 'text_clean']

    ls_tipo = []

    ls_tipo.append('classe')

    for var in X_appoggio + X_token:
        ls_tipo.append('appoggio')

    X_stilistiche = ['Adverb', 'Following_conjunction', 'Conditional', 'Strong_modal',
                     'Negation', 'Weak_modal', 'Exclamation_point',
                     'Inferential_conjunction', 'Second_person', 'Superlative',
                     'Have_form_verb', 'Proper_nouns', 'Be_form_verb', 'Third_person',
                     'Adjective', 'Other', 'Contrast_conjunction', 'Question_particle',
                     'Participle', 'Modal', 'Definit_determiners', 'Gerund', 'First_person',
                     'Past_tense']
    for var in X_stilistiche:
        ls_tipo.append('stilistiche')

    X_contenuto = ['url_count', 'termini_biomedici', 'termini_biomedici_unici',
                   'termini_commerciali', 'termini_commerciali_unici',
                   'termini_privacy', 'termini_contattaci']
    for var in X_contenuto:
        ls_tipo.append('contenuto')

    X_LIWC = ['achievement', 'adjectives', 'adverbs', 'affect', 'affiliation',
              'all_punctuation', 'anger_words', 'anxiety_words', 'apostrophes',
              'articles', 'assent', 'auxiliary_verbs', 'biological_processes', 'body',
              'causation', 'certainty', 'cognitive_processes', 'colons', 'commas',
              'comparisons', 'conjunctions', 'dashes', 'death', 'differentiation',
              'discrepancies', 'drives', 'exclamations', 'family', 'feel', 'female',
              'filler_words', 'focus_future', 'focus_past', 'focus_present',
              'friends', 'function_words', 'health', 'hear', 'home', 'i',
              'impersonal_pronouns', 'informal_language', 'ingestion', 'insight',
              'interrogatives', 'leisure', 'male', 'money', 'motion', 'negations',
              'negative_emotion_words', 'netspeak', 'nonfluencies', 'numbers',
              'other_grammar', 'other_punctuation', 'parentheses',
              'perceptual_processes', 'periods', 'personal_concerns',
              'personal_pronouns', 'positive_emotion_words', 'power', 'prepositions',
              'pronouns', 'quantifiers', 'question_marks', 'quotes', 'relativity',
              'religion', 'reward', 'risk', 'sad_words', 'see', 'semicolons',
              'sexual', 'she_he', 'social', 'space', 'swear_words', 'tentative',
              'they', 'time', 'time_orientation', 'verbs', 'we', 'work', 'you']
    for var in X_LIWC:
        ls_tipo.append('LIWC')

    X_HPNF = ['S2', 'S3', 'S4', 'S6', 'S7', 'T1', 'T2', 'T3', 'T4', 'T5',
              'T6', 'T7', 'T8', 'S10', 'S11', 'S12', 'S13', 'S14', 'T9', 'T10',
              'T11', 'L1', 'L2', 'L3', 'L4']
    # for var in X_HPNF:
    #     ls_tipo.append('hpnf')

    X_rete = X_HPNF + ['E1', 'E2', 'E3', 'E4', 'E5']
    for var in X_rete:
        ls_tipo.append('rete')

    X_utenti = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'CU1', 'CU2', 'CU3', 'CU4', 'CU5', 'CU6']
    for var in X_utenti:
        ls_tipo.append('utenti')

    # X_emozionali = [var for var in df.columns if var not in X_appoggio + X_token + X_stilistiche + X_contenuto + X_rete + X_utenti + X_LIWC + [y_column]]

    X_emozionali = ["['anger']", "['anticipation']", "['disgust']", "['fear']", "['joy']", "['sadness']",
                    "['surprise']", "['trust']", "['anger', 'anticipation']",
                    "['anger', 'disgust']", "['anger', 'fear']", "['anger', 'joy']", "['anger', 'sadness']",
                    "['anger', 'surprise']", "['anger', 'trust']",
                    "['anticipation', 'disgust']", "['anticipation', 'fear']", "['anticipation', 'joy']",
                    "['anticipation', 'sadness']",
                    "['anticipation', 'surprise']", "['anticipation', 'trust']", "['disgust', 'fear']",
                    "['disgust', 'joy']", "['disgust', 'sadness']",
                    "['disgust', 'surprise']", "['disgust', 'trust']", "['fear', 'joy']", "['fear', 'sadness']",
                    "['fear', 'surprise']", "['fear', 'trust']",
                    "['joy', 'sadness']", "['joy', 'surprise']", "['joy', 'trust']", "['sadness', 'surprise']",
                    "['sadness', 'trust']", "['surprise', 'trust']",
                    'Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'pol_score', 'pol_label', 'sub_score']

    for var in X_emozionali:
        ls_tipo.append('emozionali')

    ls_variabili = [
                       'class'] + X_appoggio + X_token + X_stilistiche + X_contenuto + X_LIWC + X_rete + X_utenti + X_emozionali

    df_tipo = pd.DataFrame({'feature': ls_variabili, 'type_information': ls_tipo})

    try:
        return list(df_tipo[df_tipo['feature'] == variabile]['type_information'])[0]
    except:
        return 'eliminare'


def merge_col(ls_col):
    df_col = pd.DataFrame(ls_col).rename(columns={0: 'col'})
    df_col['type_information'] = df_col['col'].apply(lambda x: tipo_var(x))
    df_col = df_col[df_col['type_information'] != 'eliminare']

    return list(df_col['col'])


def str_to_list(text):
    text = str(text).replace('\'', '')
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace('\"', '')
    return text.split(', ')


def concat_twitter_data(percorso):
    flist = [os.path.join(path, name) for path, subdirs, files in os.walk(percorso) for name in files]
    ls_df = []
    for file in flist:
        if 'Utenti' in percorso:
            try:
                ls_df.append(pd.read_csv(file))
            except:
                continue
        else:
            try:
                ls_df.append(pd.read_csv(file, parse_dates=['time']))
            except:
                continue
    df = pd.concat([ds for ds in ls_df])
    return df


def from_dict_to_columns(df, dict_column_name):
    df_temp = pd.io.json.json_normalize(df[dict_column_name])
    df = df.drop(columns=dict_column_name)
    # df = df.drop(columns=df_temp.columns)
    df = df.merge(df_temp, left_index=True, right_index=True)
    return df


def concat_results(path):
    flist = [os.path.join(path, name) for path, subdirs, files in os.walk(path) for name in files]
    ls_df = []
    df = pd.DataFrame()
    for file in flist:
        try:
            ls_df.append(pd.read_csv(file))
        except:
            continue
        df = pd.concat([ds for ds in ls_df])
    return df

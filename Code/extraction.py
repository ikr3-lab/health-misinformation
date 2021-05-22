from Code.feature_extraction import Emotionals, Medical, Stylistic
from Code.feature_extraction.Propagation import aggregation_twitter_data, micro, macro, parse_date_tweet
from Code.utils import Textprocessing, Processing

import pandas as pd
import spacy
import nltk

from Code.utils.Processing import str_to_list

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('universal_tagset')
nltk.download('wordnet')

# Path
absolute_path = "YOUR_ABSOLUTE_PATH"

path_data = f"{absolute_path}/Data/data_cleaned"
path_data_out = f"{absolute_path}/Data/data_w_feature"
path_resources = f"{absolute_path}/Resources"
path_data_twitter = f"{absolute_path}/Data/data_twitter"

ls_df = ['CoAID', 'FakeHealth_Release', 'ReCOVery', 'FakeHealth_Story']

for df_name in ls_df:
    print(df_name)

    df = pd.read_csv(f"{path_data}/{df_name}.csv", parse_dates=['publish_date'])

    # Text preprocessing
    df['text_clean'] = df['text'].apply(lambda x: Textprocessing.clean_text(x))
    df['token'] = df['text_clean'].apply(lambda x: nltk.tokenize.word_tokenize(x))
    df['token_clean'] = df['token'].apply(lambda x: Textprocessing.remove_stopwords(x))
    df['token_lem'] = df['token_clean'].apply(lambda x: Textprocessing.lemmatization(x))
    df['len'] = df['token_lem'].apply(lambda x: len(x))
    df = df[df['len'] > 5]
    df = df.drop(columns='len')

    # Emotional - NRC
    df_emotions = pd.read_csv(f"{path_resources}/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
                              names=["word", "emotion", "association"], sep='\t')
    df_emotions = df_emotions[df_emotions['association'] == 1]
    df_emotions = df_emotions[df_emotions['emotion'].isin(
        ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'])]

    df['dict_emotions'] = df['token'].apply(lambda x: Emotionals.emotions(df_emotions, x))
    df = Processing.from_dict_to_columns(df, dict_column_name='dict_emotions')

    # Emotional - polarity and subjectivity
    df['pol_score'] = df['text_clean'].apply(lambda x: Emotionals.polarity(x))
    df['sub_score'] = df['text_clean'].apply(lambda x: Emotionals.subjectivity(x))

    # Emotional - text2emotion
    df['emo_temp'] = df['text_clean'].apply(lambda x: Emotionals.emotions_t2e(x))
    df = Processing.from_dict_to_columns(df, dict_column_name='emo_temp')

    # Medical - linguistic
    NER_model = spacy.load(f"{path_resources}/en_core_sci_md-0.2.5/")

    df_commercials = Medical.read_ls(f"{path_resources}/commercial_terms.txt")

    df['commercial_terms_unique'] = df['token'].apply(
        lambda x: Medical.termini_count(token=x, ls_termini=df_commercials, unique=True))
    df['commercial_terms'] = df['token'].apply(
        lambda x: Medical.termini_count(token=x, ls_termini=df_commercials, unique=False))

    df['url_count'] = df['text'].apply(lambda x: Medical.count_url(x))

    df['bio'] = df['token_clean'].apply(lambda x: Medical.biomedical(NER_model, x))
    df['biomedical_terms'] = df['bio'].apply(lambda x: len(x)) / df['token_clean'].apply(lambda x: len(x))
    df['biomedical_terms_unique'] = df['bio'].apply(lambda x: len(set(x))) / df['token_clean'].apply(lambda x: len(x))
    df.drop(columns='bio', inplace=True)

    # Stylistic - linguistic
    df['dict_stylistic'] = df['token'].apply(lambda x: Stylistic.stylistic(x))
    df = Processing.from_dict_to_columns(df, dict_column_name='dict_stylistic')

    # Propagation and user profile
    if df_name != 'RecCOVery':
        df_tweet, df_users_tweet, df_retweet, df_users_retweet, df_reply, df_users_reply = aggregation_twitter_data(path_data_twitter, df_name)
    else:
        df_tweet, df_users_tweet, df_retweet, df_users_retweet = aggregation_twitter_data(path_data_twitter, df_name)

    df_tweet['time'] = df_tweet['time'].apply(parse_date_tweet)
    df_tweet['tweet'] = df_tweet['tweet'].astype(str)
    df_retweet['original_tweet'] = df_retweet['original_tweet'].astype(str)

    df['tweets'] = df['tweets'].apply(str_to_list)
    if df_name != 'ReCOVery':
        df['replies'] = df['replies'].apply(str_to_list)

    ls_macro = []
    ls_micro = []
    for index, row in df.iterrows():
        # macro (and user)
        ls_macro.append(macro(row, df_tweet, df_users_tweet, df_retweet, df_users_retweet))
        # micro (and user)
        if df_name != 'ReCOVery':
            ls_micro.append(micro(row, df_tweet, df_reply, df_users_reply))

    df['macro'] = ls_macro
    df['micro'] = ls_micro
    # From dict to columns
    df = Processing.from_dict_to_columns(df, dict_column_name='macro')
    if df_name != 'ReCOVery':
        df = Processing.from_dict_to_columns(df, dict_column_name='micro')

    # Export
    df.to_csv(f"{path_data_out}/{df_name}.csv", index=False)

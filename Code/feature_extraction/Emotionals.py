from Code.utils.Processing import str_to_list

from textblob import TextBlob
import text2emotion as te
from itertools import combinations


def polarity(text):
    return TextBlob(text).sentiment.polarity


def polarity_label(polarity):
    if polarity > 0:
        return 'Pos'
    elif polarity < 0:
        return 'Neg'
    else:
        return 'Neu'


def subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def emotions_t2e(text):
    return te.get_emotion(text)


def emotions(df_emotions, token):
    input = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    output = sum([list(map(list, combinations(input, i))) for i in range(len(input) + 1)], [])
    output = [x for x in output if len(x) <= 2 and x != []]
    dict_count = {}
    if type(token) == str:
        token = str_to_list(token)
    token = [x for x in token if len(x) > 1]
    for comb in output:
        dict_count[str(comb)] = 0
    len_token = len(token)
    for el in token:
        df_select = df_emotions[df_emotions['word'] == el]
        if df_select.shape[0] > 0:
            ls_var = list(df_select['emotion'])
            if len(ls_var) <= 2:
                ls_var.sort()
                dict_count[str(ls_var)] += 1 / len_token
    return dict_count

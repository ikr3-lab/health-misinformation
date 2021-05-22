import nltk
from collections import Counter, defaultdict

from Code.utils.Processing import str_to_list

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('universal_tagset')
nltk.download('wordnet')


def stylistic(ls_token):
    if type(ls_token) == str:
        ls_token = str_to_list(ls_token)
        ls_token = [x for x in ls_token if len(x) > 1]
    ls_sm = ['might', 'could', 'can', 'would', 'may']
    ls_wm = ['should', 'ought', 'need', 'shall', 'will']
    ls_cond = ['if']
    ls_neg = ['no', 'not', 'neither', 'nor', 'never']
    ls_inf_conj = ['therefore', 'thus', 'furthermore', 'so']
    ls_contrast_conj = ['until', 'despite', 'spite', 'though']
    ls_follow_conj = ['but', 'however', 'otherwise', 'yet', 'while', 'whilst', 'whereas']
    ls_first_pers = ['i', 'we', 'me', 'my', 'mine', 'us', 'our']
    ls_sec_pers = ['you', 'your', 'yours']
    ls_third_pers = ['he', 'she', 'him', 'her', 'his', 'it', 'its']
    ls_quest_part = ['why', 'what', 'when', 'which', 'who']
    ls_ep = ['!']
    ls_bfv = ['be', 'am', 'is', 'are', 'was', 'were', 'been', 'being']
    ls_hfv = ['have', 'has', 'had', 'having']
    ris = nltk.pos_tag(ls_token)
    ls = []
    dict_temp = {'Negation': 0, 'Contrast_conjunction': 0, 'Following_conjunction': 0, 'Inferential_conjunction': 0,
               'Exclamation_point': 0, 'Question_particle': 0, 'Be_form_verb': 0, 'Have_form_verb': 0,
               'Definit_determiners': 0,
               'Adjective': 0, 'Superlative': 0, 'Adverb': 0, 'Gerund': 0, 'Past_tense': 0, 'Participle': 0, 'Modal': 0,
               'Strong_modal': 0, 'Weak_modal': 0, 'First_person': 0, 'Second_person': 0, 'Third_person': 0,
               'Conditional': 0,
               'Proper_nouns': 0, 'Other': 0}
    for couple in ris:
        token, tag = couple[0], couple[1]
        if token in ls_neg:
            ls.append('Negation')
        elif token in ls_contrast_conj:
            ls.append('Contrast_conjunction')
        elif token in ls_follow_conj:
            ls.append('Following_conjunction')
        elif token in ls_inf_conj:
            ls.append('Inferential_conjunction')
        elif token in ls_ep:
            ls.append('Exclamation_point')
        elif token in ls_quest_part:
            ls.append('Question_particle')
        elif token in ls_bfv:
            ls.append('Be_form_verb')
        elif token in ls_hfv:
            ls.append('Have_form_verb')
        elif tag == 'DT':
            ls.append('Definit_determiners')
        elif tag == 'JJ':
            ls.append('Adjective')
        elif tag == 'JJS':
            ls.append('Superlative')
        elif tag == 'RB':
            ls.append('Adverb')
        elif tag == 'VBG':
            ls.append('Gerund')
        elif tag == 'VBD':
            ls.append('Past_tense')
        elif tag == 'VBN':
            ls.append('Participle')
        elif tag == 'MD':
            ls.append('Modal')
            if token in ls_sm:
                ls.append('Strong_modal')
            elif token in ls_wm:
                ls.append('Weak_modal')
        elif (tag == 'PRP') or (tag == 'PRP$'):
            if token in ls_first_pers:
                ls.append('First_person')
            elif token in ls_sec_pers:
                ls.append('Second_person')
            elif token in ls_third_pers:
                ls.append('Third_person')
            else:
                ls.append('Other')
        elif tag == 'IN':
            if token in ls_cond:
                ls.append('Conditional')
            else:
                ls.append('Other')
        elif (tag == 'NNP') or (tag == 'NNPS'):
            ls.append('Proper_nouns')
        else:
            ls.append('Other')
    dict_ls = {}
    for k, v in Counter(ls).items():
        dict_ls[k] = v / len(ls_token)
    dict = dsum(dict_temp, dict_ls)
    return dict


def dsum(*dicts):
    ret = defaultdict(float)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)

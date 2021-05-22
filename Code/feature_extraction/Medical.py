from Code.utils.Processing import str_to_list

import re


def read_ls(filename):
    with open(filename) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    return content


def termini_count(token, ls_termini, unique):
    if type(token) == str:
        token = str_to_list(token)
    token = [x for x in token if len(x) > 1]
    if not unique:
        return len([t for t in token if t in ls_termini]) / len(token)
    else:
        return len(set([t for t in token if t in ls_termini])) / len(token)


def count_url(text):
    url_a = len(re.findall(r"(https?://\S+)", text))
    url_b = len(re.findall(r"(www)", text))
    return url_a + url_b


def biomedical(ner_model, token):
    if type(token) == str:
        token = str_to_list(token)
    token = [x for x in token if len(x) > 1]
    text = ' '.join(token)
    doc = ner_model(text)
    list_biomedical = [str(x).lower() for x in list(doc.ents)]
    return list_biomedical
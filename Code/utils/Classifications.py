from keras.preprocessing.text import Tokenizer
from keras import regularizers
from keras.layers import Input, Embedding, Flatten, Dense, \
    Conv1D, MaxPooling1D, LSTM, Bidirectional, concatenate
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model

import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

from Code.feature_selection import CFS


def missing_values(df):
    return df.replace([-1], 999)


def metrics_class(dict_metrics):
    precision_neg = dict_metrics['0']['precision']
    precision_pos = dict_metrics['1']['precision']
    recall_neg = dict_metrics['0']['recall']
    recall_pos = dict_metrics['1']['recall']
    try:
        f1_score_neg = (2 * precision_neg * recall_neg) / (precision_neg + recall_neg)
    except:
        f1_score_neg = 0
    try:
        f1_score_pos = (2 * precision_pos * recall_pos) / (precision_pos + recall_pos)
    except:
        f1_score_pos = 0
    return {'Precision +': precision_pos, 'Precision -': precision_neg, 'Recall +': recall_pos, 'Recall -': recall_neg,
            'F1 +': f1_score_pos, 'F1 -': f1_score_neg}


def include_features(stylistic=False, emotional=False, medical=False, propagation=False, user=False, hpnf=False, liwc=False,
                     token=False):
    x_columns = []

    x_token = ['token', 'token_clean', 'token_lem', 'text_clean']

    x_stylistic = ['Adverb', 'Following_conjunction', 'Conditional', 'Strong_modal',
                   'Negation', 'Weak_modal', 'Exclamation_point',
                   'Inferential_conjunction', 'Second_person', 'Superlative',
                   'Have_form_verb', 'Proper_nouns', 'Be_form_verb', 'Third_person',
                   'Adjective', 'Other', 'Contrast_conjunction', 'Question_particle',
                   'Participle', 'Modal', 'Definit_determiners', 'Gerund', 'First_person',
                   'Past_tense']

    x_medical = ['url_count', 'biomedical_terms', 'biomedical_terms_unique',
                 'commercial_terms', 'commercial_terms_unique']

    x_hpnf = ['S2', 'S3', 'S4', 'S6', 'S7', 'T1', 'T2', 'T3', 'T4', 'T5',
              'T6', 'T7', 'T8', 'S10', 'S11', 'S12', 'S13', 'S14', 'T9', 'T10',
              'T11', 'L1', 'L2', 'L3', 'L4']

    x_propagation = x_hpnf + ['E1', 'E2', 'E3', 'E4', 'E5']

    x_user = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'CU1', 'CU2', 'CU3', 'CU4', 'CU5', 'CU6']

    x_emotional = ["['anger']", "['anticipation']", "['disgust']", "['fear']", "['joy']", "['sadness']",
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
                   'Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'pol_score', 'sub_score']

    x_liwc = ['achievement', 'adjectives', 'adverbs', 'affect', 'affiliation',
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

    if stylistic:
        x_columns += x_stylistic

    if medical:
        x_columns += x_medical

    if propagation:
        x_columns += x_propagation

    if user:
        x_columns += x_user

    if emotional:
        x_columns += x_emotional

    if token:
        x_columns += x_token

    if hpnf:
        x_columns += x_hpnf

    if liwc:
        x_columns += x_liwc

    return x_columns


def feature_type(feature):
    x_temp = ['news_id', 'publish_date', 'text', 'tweets', 'retweets', 'replies', 'len', 'emoction', 'len_text']

    x_token = ['token', 'token_clean', 'token_lem', 'text_clean']

    ls_type = ['class']

    for var in x_temp + x_token:
        ls_type.append('temp')

    x_stylistic = ['Adverb', 'Following_conjunction', 'Conditional', 'Strong_modal',
                   'Negation', 'Weak_modal', 'Exclamation_point',
                   'Inferential_conjunction', 'Second_person', 'Superlative',
                   'Have_form_verb', 'Proper_nouns', 'Be_form_verb', 'Third_person',
                   'Adjective', 'Other', 'Contrast_conjunction', 'Question_particle',
                   'Participle', 'Modal', 'Definit_determiners', 'Gerund', 'First_person',
                   'Past_tense']
    for var in x_stylistic:
        ls_type.append('stylistic')

    x_medical = ['url_count', 'biomedical_terms', 'biomedical_terms_unique',
                 'commercial_terms', 'commercial_terms_unique']
    for var in x_medical:
        ls_type.append('medical')

    x_propagation = ['S2', 'S3', 'S4', 'S6', 'S7', 'T1', 'T2', 'T3', 'T4', 'T5',
                     'T6', 'T7', 'T8', 'S10', 'S11', 'S12', 'S13', 'S14', 'T9', 'T10',
                     'T11', 'L1', 'L2', 'L3', 'L4', 'E1', 'E2', 'E3', 'E4', 'E5']
    for var in x_propagation:
        ls_type.append('propagation')

    x_user = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10', 'CU1', 'CU2', 'CU3', 'CU4', 'CU5', 'CU6']
    for var in x_user:
        ls_type.append('user')

    x_emotional = ["['anger']", "['anticipation']", "['disgust']", "['fear']", "['joy']", "['sadness']",
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
    for var in x_emotional:
        ls_type.append('emotional')

    ls_features = ['class'] + x_temp + x_token + x_stylistic + x_medical + x_propagation + x_user + x_emotional

    df_type = pd.DataFrame({'feature': ls_features, 'type_information': ls_type})

    try:
        return list(df_type[df_type['feature'] == feature]['type_information'])[0]
    except:
        return 'delete'


def identity_tokenizer(text):
    return text


def vectorizer():
    tfidf = TfidfVectorizer(
        analyzer='word',
        tokenizer=identity_tokenizer,
        preprocessor=identity_tokenizer,
        token_pattern=None)

    binary = CountVectorizer(
        analyzer='word',
        tokenizer=identity_tokenizer,
        preprocessor=identity_tokenizer,
        binary=True,
        token_pattern=None)

    return {'binari': binary, 'tfidf': tfidf}


def fsk(X_train, X_test, y_train):
    mat_train = X_train
    mat_test = X_test
    kb = SelectKBest(chi2, k=750)
    kb.fit_transform(mat_train, y_train)
    train = pd.DataFrame(kb.fit_transform(mat_train, y_train))
    test = pd.DataFrame(kb.transform(mat_test))
    return train, test


def embedding_index(glove_path, embedding_dim):
    embeddings_index = {}
    f = open(f"{glove_path}/glove.6B.{embedding_dim}d.txt", encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def df_word_embedding_mean(df, embeddings_index, embedding_dim):
    df['word_embedding_mean'] = df['token_clean'].apply(lambda x: word_embedding_mean(x, embeddings_index))
    for i in range(embedding_dim):
        tmp = []
        for index, row in df.iterrows():
            tmp.append(row['word_embedding_mean'][i])
        df[str(i)] = tmp
    return df


def word_embedding_mean(token_clean, embeddings_index):
    sum = 0
    count = 0
    for token in token_clean:
        if token in embeddings_index.keys():
            sum += embeddings_index[token]
            count += 1
    return sum / count


def model_cnn(embedding_layer):
    sequence_input = Input(shape=(1000,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(5)(l_cov2)
    # l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
    # l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(32, activation='relu')(l_flat)
    # x = Dense(256, activation='relu')(l_dense)
    preds = Dense(2, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)

    return model


def model_lstm(embedding_layer):
    seq_length = 1000
    nlp_input = Input(shape=(seq_length,), name='nlp_input')
    emb = embedding_layer(nlp_input)
    nlp_out = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=regularizers.l2(0.01)))(
        emb)
    x = Dense(2, activation='sigmoid')(nlp_out)
    model = Model(inputs=[nlp_input], outputs=[x])

    return model


def model_cnn_all_features(dim_meta_input, emb_layer):
    nlp_input = Input(shape=(1000,), dtype='int32')
    meta_input = Input(shape=(dim_meta_input,), name='meta_input')
    embedded_sequences = emb_layer(nlp_input)
    l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(5)(l_cov2)
    # l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
    # l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
    l_flat = Flatten()(l_pool2)
    # l_dense = Dense(128, activation='relu')(l_flat)
    # meta_input_1 = Dense(64, activation='relu')(meta_input)
    x = concatenate([l_flat, meta_input])
    x = Dense(32, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=[nlp_input, meta_input], outputs=[x])

    return model


def model_lstm_all_features(dim_meta_input, emb_layer):
    nlp_input = Input(shape=(1000,), name='nlp_input')
    meta_input = Input(shape=(dim_meta_input,), name='meta_input')
    emb = emb_layer(nlp_input)
    nlp_out = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.2, kernel_regularizer=regularizers.l2(0.01)))(
        emb)
    x = concatenate([nlp_out, meta_input])
    x = Dense(32, activation='relu')(nlp_out)
    x = Dense(2, activation='sigmoid')(x)
    model = Model(inputs=[nlp_input, meta_input], outputs=[x])

    return model


# Embedding layer
def glove_embedding(df, x_train, x_test, y_train, y_test, embedding_dim):
    max_sequence_length = 1000
    max_nb_words = 20000

    texts = df['text_clean']
    sentences = texts.apply(lambda x: x.split())
    tokenizer = Tokenizer(num_words=max_nb_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=max_sequence_length)
    labels = to_categorical(np.asarray(y_train.values))

    x_train = data[list(x_train.index)]
    y_train = to_categorical(np.asarray(y_train.values))
    x_val = data[list(x_test.index)]
    y_val = to_categorical(np.asarray(y_test.values))

    path_resources = "C:\\Users\Stefano\Desktop\Tesi\Resources"
    embeddings_index = embedding_index(f"{path_resources}/Glove", embedding_dim=embedding_dim)

    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=True)

    return x_train, x_val, y_train, y_val, embedding_layer



def feature_selection_classification(X_train, y_train):
    ls_feature = []
    diz_col = {}
    dict_type = {}

    for col in pd.DataFrame(X_train).columns:
        diz_col[col] = feature_type(col)
    for col, type in diz_col.items():
        if len(set(pd.DataFrame(X_train)[col])) > 1:
            if type not in dict_type:
                dict_type[type] = [col]
            else:
                dict_type[type].append(col)

    for type, cols in dict_type.items():
        print('Features type ' + str(type) + ' ' + str(len(cols)))
        fcs_fin = CFS.cfs(np.array(X_train.loc[:, cols]), np.array(y_train))
        ls_feature += list((X_train.iloc[:, fcs_fin]).columns)

    return ls_feature

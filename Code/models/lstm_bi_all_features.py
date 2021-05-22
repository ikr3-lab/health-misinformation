import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

from sklearn import preprocessing
from Code.utils.Classifications import embedding_index, metrics_class, glove_embedding, \
     include_features, model_lstm_all_features, feature_selection_classification
from Code.utils.Processing import str_to_list

np.random.seed(123)
skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)

absolute_path = sys.argv[1]

path_data = f"{absolute_path}/Data/data_w_feature"
path_data_out = f"{absolute_path}/Data/results/global"
path_resources = f"{absolute_path}/Resources"

# Embedding GloVe
embeddings_index = embedding_index(f"{path_resources}/Glove", embedding_dim=100)

ls_df = ['CoAID', 'FakeHealth_Release', 'ReCOVery', 'FakeHealth_Story']

for df_name in ls_df:
    print(df_name)

    df_temp = pd.DataFrame()

    df = pd.read_csv(f"{path_data}/{df_name}.csv")

    df['token'] = df['token'].apply(str_to_list)
    df['token_clean'] = df['token_clean'].apply(str_to_list)
    df['token_lem'] = df['token_lem'].apply(str_to_list)

    x_col = include_features(stylistic=True, emotional=True, medical=True, propagation=True, user=True)
    x_col = [col for col in x_col if col in df.columns]
    X, y = df[x_col], df['class']

    # Text clean per Word Embedding
    X_columns_name = ['text_clean']
    y_name = ['class']
    X_nlp, y_nlp = df[X_columns_name], df[y_name]

    skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)


    for train_index, test_index in skf.split(X, y):

        # Embedding
        X_train_nlp, X_test_nlp = X_nlp.iloc[train_index, :], X_nlp.iloc[test_index, :]
        y_train_nlp, y_test_nlp = y_nlp.iloc[train_index, :], y_nlp.iloc[test_index, :]

        # Temp
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        # Feature selection
        ls_feature = feature_selection_classification(X_train, y_train)
        X_train = X_train.loc[:, ls_feature]
        X_test = X_test.loc[:, ls_feature]

        # Normalization
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        ls_emb_dim = [50, 100, 200]
        for emb_dim in ls_emb_dim:

            # Embedding layer
            X_train_e, X_test_e, y_train_e, y_test_e, emb_layer = glove_embedding(df, X_train_nlp, X_test_nlp, y_train_nlp, y_test_nlp, emb_dim)

            model = model_lstm_all_features(X_train.shape[1], emb_layer)

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=[tf.keras.metrics.AUC()])
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
            mc = ModelCheckpoint(f"{path_resources}/best_model.h5", monitor='val_loss', mode='min', verbose=0,
                                 save_best_only=True)

            model.fit([X_train_e, X_train], y_train_e, validation_data=([X_test_e, X_test], y_test_e), batch_size=256,
                      epochs=15, callbacks=[es, mc], verbose=0)

            model = tf.keras.models.load_model(f"{path_resources}/best_model.h5")

            # predictions
            y_pred = model.predict([X_test_e, X_test])

            # Metrics
            dict_metrics_class = metrics_class(
                classification_report(np.argmax(y_test_e, axis=1), np.argmax(y_pred, axis=1), output_dict=True))
            dict_acc_auc = {'Accuracy': accuracy_score(np.argmax(y_test_e, axis=1), np.argmax(y_pred, axis=1)),
                            'AUC': roc_auc_score(y_test_e, y_pred)}
            dict_metrics = {**dict_acc_auc, **dict_metrics_class}

            # Output
            dict_names = {'Dataframe': df_name, 'Classificatore': f'Bi-LSTM-all-{emb_dim}', 'Modello': 'Bi-LSTM(WE+all)'}
            df_ris = pd.DataFrame({**dict_names, **dict_metrics}, index=[0])
            df_temp = pd.concat([df_temp, df_ris])

    # export
    df_temp.to_csv(f"{path_data_out}/Bi-LSTM(WE+all)_{df_name}.csv", index=False)

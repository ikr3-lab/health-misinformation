import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

from Code.utils.Classifications import embedding_index, metrics_class, glove_embedding, model_lstm
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

    y_name = 'class'
    X_columns_name = 'token_clean'

    X, y = df[X_columns_name], df[y_name]
    skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
    ls_emb_dim = [50, 100, 200]
    for emb_dim in ls_emb_dim:
        df_temp = pd.DataFrame()
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train_e, X_test_e, y_train_e, y_test_e, emb_layer = glove_embedding(df, X_train, X_test, y_train,
                                                                                  y_test, embedding_dim=emb_dim)

            model = model_lstm(emb_layer)
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=[tf.keras.metrics.AUC()])
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
            mc = ModelCheckpoint(f"{path_resources}/best_model.h5", monitor='val_loss', mode='min', verbose=0,
                                 save_best_only=True)

            model.fit(X_train_e, y_train_e, validation_data=(X_test_e, y_test_e), epochs=15, batch_size=256,
                      callbacks=[es, mc],
                      verbose=0)

            model = tf.keras.models.load_model(f"{path_resources}/best_model.h5")

            # predictions
            y_pred = model.predict(X_test_e)

            # Metrics
            dict_metrics_class = metrics_class(
                classification_report(np.argmax(y_test_e, axis=1), np.argmax(y_pred, axis=1), output_dict=True))
            dict_acc_auc = {'Accuracy': accuracy_score(np.argmax(y_test_e, axis=1), np.argmax(y_pred, axis=1)),
                            'AUC': roc_auc_score(y_test_e, y_pred)}

            dict_metrics = {**dict_acc_auc, **dict_metrics_class}

            # Output
            dict_names = {'Dataframe': df_name, 'Classificatore': f'LSTM-{emb_dim}', 'Modello': 'Bi-LSTM(WE)'}
            df_ris = pd.DataFrame({**dict_names, **dict_metrics}, index=[0])
            df_temp = pd.concat([df_temp, df_ris])

        # export
        df_temp.to_csv(f"{path_data_out}/Bi-LSTM(WE)_{df_name}_{emb_dim}.csv", index=False)

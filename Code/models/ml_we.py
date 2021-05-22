from sklearn import preprocessing

from Code.utils.Classifications import missing_values, metrics_class, include_features, embedding_index, \
    df_word_embedding_mean, feature_selection_classification

import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from Code.utils.Processing import str_to_list
import sys

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

    # Embedding:
    df_we = df_word_embedding_mean(df, embeddings_index=embeddings_index, embedding_dim=100)
    df_we = df_we.drop(axis=0, columns='word_embedding_mean')

    dict_classifiers = {'Random Forest': RandomForestClassifier(),
                        'Reg Log': LogisticRegression(),
                        'Naive Bayes': GaussianNB(),
                        'GBC': GradientBoostingClassifier()}

    y_col = 'class'

    x_col = include_features(stylistic=True, emotional=True, medical=True, propagation=True, user=True)
    x_col = [col for col in x_col if col in df.columns] + [str(i) for i in range(100)] # adding 100 word embedding features
    X, y = df_we[x_col], df_we[y_col]

    # Missing
    X = missing_values(X)

    for train_index, test_index in skf.split(X, y):

        # Train - test all features
        X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(test_index)]
        y_train, y_test = y[y.index.isin(train_index)], y[y.index.isin(test_index)]

        # Train - test only word embedding features
        X_train_we = X_train[[str(i) for i in range(100)]]
        X_test_we = X_test[[str(i) for i in range(100)]]

        # Train - test without word embedding features
        X_train = X_train.drop(columns=([str(i) for i in range(100)]))
        X_test = X_test.drop(columns=([str(i) for i in range(100)]))

        # Feature selection
        ls_feature = feature_selection_classification(X_train, y_train)
        X_train = X_train.loc[:, ls_feature]
        X_test = X_test.loc[:, ls_feature]

        # Normalization
        scaler = preprocessing.MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Join features and word embedding features
        X_train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_train_we).reset_index(drop=True)], axis=1)
        X_test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(X_test_we).reset_index(drop=True)], axis=1)

        for clf_name, clf in dict_classifiers.items():

            clf.fit(X_train, y_train)

            # predictions
            y_pred = clf.predict(X_test)

            # Metrics
            dict_metrics_class = metrics_class(
                classification_report(y_test, y_pred, output_dict=True))
            dict_acc_auc = {'Accuracy': accuracy_score(y_test, y_pred),
                            'AUC': roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])}

            dict_metrics = {**dict_acc_auc, **dict_metrics_class}

            # Output
            dict_names = {'Dataframe': df_name, 'Classificatore': clf_name, 'Modello': 'ML(WE+all)'}
            df_ris = pd.DataFrame({**dict_names, **dict_metrics}, index=[0])
            df_temp = pd.concat([df_temp, df_ris])

        # export
        df_temp.to_csv(f"{path_data_out}\ML(WE+all)_{df_name}.csv", index=False)

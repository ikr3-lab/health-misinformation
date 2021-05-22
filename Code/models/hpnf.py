from Code.utils.Classifications import missing_values, metrics_class, include_features

import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import sys

np.random.seed(123)
skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)

absolute_path = sys.argv[1]

path_data = f"{absolute_path}\Data\data_w_feature"
path_data_out = f"{absolute_path}/Data/results/global"

ls_df = ['CoAID', 'FakeHealth_Release', 'ReCOVery', 'FakeHealth_Story']

for df_name in ls_df:
    print(df_name)

    df_temp = pd.DataFrame()

    df = pd.read_csv(f"{path_data}/{df_name}.csv")

    dict_classifiers = {'Random Forest': RandomForestClassifier(),
                        'Reg Log': LogisticRegression(),
                        'Naive Bayes': GaussianNB(),
                        'GBC': GradientBoostingClassifier()}

    y_col = 'class'

    x_col = include_features(hpnf=True)
    x_col = [col for col in x_col if col in df.columns]
    X, y = df[x_col], df[y_col]

    # Missing
    X = missing_values(X)

    for train_index, test_index in skf.split(X, y):

        # Train - test
        X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(test_index)]
        y_train, y_test = y[y.index.isin(train_index)], y[y.index.isin(test_index)]

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
            dict_names = {'Dataframe': df_name, 'Classificatore': clf_name, 'Modello': 'hpnf'}
            df_ris = pd.DataFrame({**dict_names, **dict_metrics}, index=[0])
            df_temp = pd.concat([df_temp, df_ris])

        # export
        df_temp.to_csv(f"{path_data_out}/HPNF_{df_name}.csv", index=False)

from Code.feature_selection import CFS
from Code.utils.Classifications import missing_values, metrics_class, include_features

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from Code.utils.Processing import concat_results

np.random.seed(123)
skf = StratifiedKFold(n_splits=5, shuffle=False)

absolute_path = "YOUR_ABSOLUTE_PATH"

def class_evaluation(class_name, stylistic=False, emotional=False, medical=False, propagation=False, user=False):
    path_data = f"{absolute_path}/Data/data_w_feature"
    path_data_out = f"{absolute_path}/Data/results/feature_class"

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

        x_col = include_features(stylistic=stylistic, emotional=emotional, medical=medical, propagation=propagation,
                                 user=user)
        x_col = [col for col in x_col if col in df.columns]
        X, y = df[x_col], df[y_col]

        # Missing
        X = missing_values(X)

        for train_index, test_index in skf.split(X, y):

            # Train - test
            X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(test_index)]
            y_train, y_test = y[y.index.isin(train_index)], y[y.index.isin(test_index)]

            # Feature selection
            features_ok = CFS.cfs(np.array(X_train), np.array(y_train))
            X_train = X_train.iloc[:, features_ok]
            X_test = X_test.iloc[:, features_ok]

            # Normalizer
            scaler = preprocessing.MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

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
                dict_names = {'Dataframe': df_name, 'Classificatore': clf_name, 'Modello': class_name}
                df_ris = pd.DataFrame({**dict_names, **dict_metrics}, index=[0])
                df_temp = pd.concat([df_temp, df_ris])

            # export
            df_temp.to_csv(f"{path_data_out}/{class_name}_{df_name}.csv", index=False)


# Evaluation
class_evaluation('stylistic', stylistic=True, emotional=False, medical=False, propagation=False, user=False)
class_evaluation('emotional', stylistic=False, emotional=True, medical=False, propagation=False, user=False)
class_evaluation('medical', stylistic=False, emotional=False, medical=True, propagation=False, user=False)
class_evaluation('propagation', stylistic=False, emotional=False, medical=False, propagation=True, user=False)
class_evaluation('user', stylistic=False, emotional=False, medical=False, propagation=False, user=True)


# Results
df = concat_results(f"{absolute_path}/Data/results/feature_class/")

metrics = ['AUC']

df_mean = df.groupby(['Dataframe', 'Modello', 'Classificatore']).mean()

print(round(df_mean.groupby(['Dataframe', 'Modello'])[metrics].max().sort_values(['Dataframe','AUC','Modello'], ascending=False), 3))

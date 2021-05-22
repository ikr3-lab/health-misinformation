import pandas as pd

from Code.utils.Processing import concat_results

absolute_path = "YOUR_ABSOLUTE_PATH"

path_data = f"{absolute_path}/Data/results/global/"

df = concat_results(path_data)

# Aggregate F-measure
ls_df = ['CoAID', 'FakeHealth_Release', 'ReCOVery', 'FakeHealth_Story']
dict_num_class = {}
for df_name in ls_df:
    df_c = pd.read_csv(f"{absolute_path}/Data/data_w_feature/{df_name}.csv")
    dict_num_class[df_name] = {'0': df_c[df_c['class'] == 0].shape[0], '1': df_c[df_c['class'] == 1].shape[0]}

ls_f = []
for index, row in df.iterrows():
    ls_f.append(((row['F1 +'] * dict_num_class[row['Dataframe']]['1']) + (
                row['F1 -'] * dict_num_class[row['Dataframe']]['0'])) / (
                                   dict_num_class[row['Dataframe']]['1'] + dict_num_class[row['Dataframe']]['0']))
df['F-measure'] = ls_f


# means of metrics
metrics = ["AUC", "F-measure"]
df_means = df.groupby(['Dataframe', 'Modello', 'Classificatore'])[metrics].mean().reset_index()

# Best config for each model
best_idx = df_means.groupby(['Dataframe', 'Modello'])['AUC'].idxmax()

# Results order by AUC
df_r = df_means.iloc[best_idx, :][['Dataframe', 'Modello'] + metrics].sort_values(['Dataframe', 'AUC'], ascending=False)
print(df_r)

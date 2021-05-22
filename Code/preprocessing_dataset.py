from Code.utils import Processing

import pandas as pd

# Path
absolute_path = "YOUR_ABSOLUTE_PATH"
path_data = f"{absolute_path}/Data/data_raw"
path_data_out = f"{absolute_path}/Data/data_cleaned"

ls_df = ['CoAID', 'FakeHealth', 'ReCOVery']

for df_name in ls_df:

    print(df_name)

    df = pd.read_csv(f"{path_data}/{df_name}/{df_name}_raw.csv")

    if df_name == 'FakeHealth':
        df['class'] = df['criteria'].apply(lambda x: Processing.ground_truth_fakehealth(x))
        df['publish_date'] = df['publish_date'].apply(Processing.date_conversion_fakehealth)
    elif df_name == 'CoAID':
        df = Processing.select_type_coaid(df)
        df['publish_date'] = df['publish_date'].apply(Processing.date_conversion_coaid_recovery)

    # Rename columns and drop unnecessary columns
    cols_rename, cols_keep = Processing.uniform_variables_name(df_name)
    df = df[cols_keep]
    df = df.rename(columns=cols_rename)

    # Remove information without text and duplicates
    df = df[~df['text'].isnull()]
    df = df.drop_duplicates(subset='text')

    # Split FakeHealth and save dataset
    if df_name == 'FakeHealth':
        df_release, df_story = Processing.split_fakehealth(df)
        df_release.to_csv(f"{path_data_out}/{df_name}_Release.csv", index=False)
        df_story.to_csv(f"{path_data_out}/{df_name}_Story.csv", index=False)
    else:
        df.to_csv(f"{path_data_out}/{df_name}.csv", index=False)

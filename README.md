# Data

Original data are on _GitHub_, more precisely:
- **CoAID**: https://github.com/cuilimeng/CoAID
- **FakeHealth**: https://github.com/EnyanDai/FakeHealth
- **ReCOVery**: https://github.com/apurvamulay/ReCOVery

Due to _Twitter_ policy it is not possible to release data in disaggregate form.

Inside the folder `Data/data_w_feature` is possible to find the data with the points associated with each feature described in the article.



# External resources
In order to execute the code it's necessary to download GloVe pre-trained vector available here http://nlp.stanford.edu/data/glove.6B.zip and put the files insie the folder ```Resources/Glove```



# Code

### Import:
- Inside the script ```Code/import_dataset.py``` enter the proper ```absolute_path``` (ie the pathfrom _C_ to _project-name_), then execute it.

After the execution, data are placed in `Data/data_row`.

### Pre-processing
- Inside the script ```Code/preprocessing_dataset.py``` enter the proper ```absolute_path```, then execute it.

After the execution, data are placed in `Data/data_cleaned`.

### Enrichment

In order to obtain Twitter data it's necessary to make call to API _Twitter_:
- Enter the _Twitter_ keys  inside ```Code/data_enrichment/secret.json`.
- Inside the script ```Code/twitter_enrichment.py``` enter the proper ```absolute_path```, then execute it.
After the execution, data are placed in `Data/data_twitter`.

### Feature extraction

- Inside the script ```Code/extraction.py``` enter the proper ```absolute_path```, then execute it.
After the execution, data are placed in `Data/data_w_feature`.

### Class evaluation

- Inside the script ```Code/feature_class_evaluation.py``` enter the proper ```absolute_path```, then execute it.

### Global evaluation

- In order to obtain LIWC feature it's necessary enter the _Receptivity_ keys inside ```Code/data_enrichment/liwc_features.py```, set the value of `extract_liwc` to True (line 17) and then execute the script.
    - However, the previous it's not mandatory because inside `Data/data_w_feature` are already present data with these features.  
- Inside the script ```Code/global_evaluation.py``` enter the proper ```absolute_path```, then execute it (results are placed in `Data/results`).
- Inside the script ```Code/results.py``` enter the proper ```absolute_path```, then execute it.



# Some recommendations
- Due to policy _Twitter_ it's not possible make more than 180 API call per 15 minutes. Because of this the enrichment phase is pretty long.
- To execute the global evaluation It's recommended to use GPU, especially for Deep Learning models.
- If it's not necessary, in order to save a lot of time, it's recommended to use directly data placed inside the folder `Data/data_w_feature`.

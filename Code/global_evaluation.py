import os

absolute_path = 'YOUR_ABSOLUTE_PATH'

os.system("%s %s" % (absolute_path + '/Code/models/hpnf.py', absolute_path))
os.system("%s %s" % (absolute_path + '/Code/models/cnn_all_features.py', absolute_path))
os.system("%s %s" % (absolute_path + '/Code/models/cnn_we.py', absolute_path))
os.system("%s %s" % (absolute_path + '/Code/models/liwc.py', absolute_path))
os.system("%s %s" % (absolute_path + '/Code/models/lstm_di_all_features.py', absolute_path))
os.system("%s %s" % (absolute_path + '/Code/models/lstm_bi_we.py', absolute_path))
os.system("%s %s" % (absolute_path + '/Code/models/ml_bow.py', absolute_path))
os.system("%s %s" % (absolute_path + '/Code/models/ml_we.py', absolute_path))

import os

absolute_path = 'YOUR_ABSOLUTE_PATH'

os.system("%s %s" % (absolute_path + '/Code/import_dataset/import_coaid.py', absolute_path))
os.system("%s %s" % (absolute_path + '/Code/import_dataset/import_fakehealth.py', absolute_path))
os.system("%s %s" % (absolute_path + '/Code/import_dataset/import_recovery.py', absolute_path))

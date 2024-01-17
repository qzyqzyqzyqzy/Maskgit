import os
import shutil
datapath='./owndataset1/class1'
topath='./owndataset2/class2'
for root, dirs, files in os.walk(datapath):
    for file in files:
        if file.startswith('image'):
            old_path = os.path.join(root, file)
            new_path = os.path.join(topath, file)
            print(old_path)
            print(new_path)
            shutil.move(old_path, new_path)
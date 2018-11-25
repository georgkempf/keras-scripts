

import os
import random
from shutil import copyfile

dataset_dir = 'C:/Users/Saturn/cats_dogs'
output_dir = 'C:/Users/Saturn/cats_dogs_sorted'
val_frac = 0.2



folders = [x for x in os.listdir(dataset_dir)]

new_folders = ['train','validation']
for new_folder in new_folders:
    for folder in folders:
        target = os.path.join(output_dir,new_folder,folder)
        if not os.path.exists(target):
            os.makedirs(target)



for folder in folders:
    files = [os.path.join(dataset_dir,folder,x) for x in os.listdir(os.path.join(dataset_dir,folder))]
    file_num = len(files)
    print("Number of files in %s: %s" %(folder,file_num))
    val_num = 0.2 * float(file_num)
    val_files = random.sample(files,int(val_num))
    train_files = [x for x in files if not x in val_files]
    for file in val_files:
        target_file = os.path.join(output_dir,'validation',folder,os.path.basename(file))
        print("Copying %s to %s" %(file, target_file))
        copyfile(file,target_file)
    for file in train_files:
        target_file = os.path.join(output_dir,'train',folder,os.path.basename(file))
        print("Copying %s to %s" %(file, target_file))
        copyfile(file,target_file)
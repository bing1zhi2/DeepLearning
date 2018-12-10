import os
import numpy as np
import h5py
m = {'a':1,'b':2}
for key, value in m.items():
    print(key,value)

filename_path="F:\\work\\runspace\\log\\20181128-171431\\stat.h5"
with h5py.File(filename_path, 'r') as f:
    distance_to_center = np.array(f.get('distance_to_center'))
    label_list = np.array(f.get('label_list'))
    image_list = np.array(f.get('image_list'))
    print(distance_to_center)





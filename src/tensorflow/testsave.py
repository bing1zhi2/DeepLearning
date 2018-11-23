import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import os

model_dir='F:/dataset/facenet/model/20180408-102900'

checkpoint_path = os.path.join(model_dir, "model-20180402-114759.ckpt-275")
print(checkpoint_path)
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))


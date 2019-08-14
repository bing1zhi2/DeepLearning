import argparse
import pdb
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes


input_model = 'F:\\dataset\\facenet\\model\\20180402-114759\\20180402-114759.pb'
input_model = 'F:\\dataset\\pre_train_model\\wdsr\\saved_model.pb'
input_model = 'F:\\code\\mycode\\DeepLearning\\src\\tensorflow_pra\\serving\\temp\\1\saved_model.pb'

def cpu2mlu(graph_def):
    for node in graph_def.node:
        print("--------------------", node.name, ", ", node.op)


with tf.gfile.FastGFile(input_model,'rb') as f:
    new_g = graph_pb2.GraphDef()
    new_g.ParseFromString(f.read())
    tf.import_graph_def(new_g)

graph_def = cpu2mlu(new_g)
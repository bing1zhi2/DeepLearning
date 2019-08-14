import argparse
import pdb
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes


input_model_dir = 'F:\\code\\mycode\\DeepLearning\\src\\tensorflow_pra\\serving\\temp\\1'

def cpu2mlu(graph_def):
    for node in graph_def.node:
        print("--------------------", node.name, ", ", node.op)

def read_new():
    with tf.Session(graph=tf.Graph()) as sess:
        metagraph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], input_model_dir)
        signature_def = metagraph_def.signature_def[
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        # input_tensor = sess.graph.get_tensor_by_name(
        #     signature_def.inputs['inputs'].name)
        # output_tensor = sess.graph.get_tensor_by_name(
        #     signature_def.outputs['output'].name)
        graph_def = sess.graph_def
        for node in graph_def.node:
            print("--------------------", node.name, ",   ", node.op)



# with tf.gfile.FastGFile(input_model,'rb') as f:
#     new_g = graph_pb2.GraphDef()
#     new_g.ParseFromString(f.read())
#     tf.import_graph_def(new_g)

# graph_def = cpu2mlu(new_g)

read_new()
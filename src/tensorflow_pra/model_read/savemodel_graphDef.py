"""
因想转换节点中的设备，把savemodel 格式的模型 转化成 冻结的pb 模型 convert_variables_to_constants

"""
import argparse
import pdb
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from PIL import Image
import numpy as np
import os


input_model_dir = 'F:\\dataset\\pre_train_model\\wdsr'
OP_NOT_SUPPORTED = ["BiasAdd"
                    ]
def read_write_new():
    with tf.Session(graph=tf.Graph()) as sess:
        metagraph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], input_model_dir)
        signature_def = metagraph_def.signature_def[
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        # input_tensor = sess.graph.get_tensor_by_name(
        #     signature_def.inputs['inputs'].name)
        # output_tensor = sess.graph.get_tensor_by_name(
        #     signature_def.outputs['output'].name)

        graph_def =metagraph_def.graph_def
        for node in graph_def.node:
            if node.op in OP_NOT_SUPPORTED:
                node.device = '/device:CPU:0'
            print("--------------------", node.name, ", ", node.op, ", ", node.device)

        GraphDef = convert_variables_to_constants(sess,graph_def,['clip_by_value'])

        with tf.gfile.FastGFile('wdsr_value.pb', "w") as f:
            f.write(GraphDef.SerializeToString())
            f.close()


def test_read():
    with tf.Session() as sess:
        with gfile.FastGFile("wdsr_value.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            print(graph_def)
            tf.import_graph_def(graph_def, name='')
            input_tensor = sess.graph.get_tensor_by_name('input_tensor:0')
            output_tensor = tf.get_default_graph().get_tensor_by_name("clip_by_value:0")

            input_file = "G:\\dataset\\Set5_bicubic\\bird.png"
            output_dir = "G:\\dataset\\testwdsr"
            output_file = os.path.join(output_dir, 'out.png')
            input_image = np.asarray(Image.open(input_file))

            def forward_images(images):
                images = images.astype(np.float32) / 255.0
                images = output_tensor.eval(feed_dict={input_tensor: images})
                return images

            input_images = np.expand_dims(input_image, axis=0)
            output_images = forward_images(input_images)
            output_image = output_images[0]

            output_image = np.around(output_image * 255.0).astype(np.uint8)
            output_image = Image.fromarray(output_image, 'RGB')
            output_image.save(output_file)




def test_prdict():


    with tf.Graph().as_default():
        with tf.Session() as sess:
            with tf.gfile.FastGFile('test2.pb', 'rb') as f:
                graph_def = graph_pb2.GraphDef()
                graph_def.ParseFromString(f.read())
                print(graph_def)
                tf.import_graph_def(graph_def)

                # Get input and output tensors
                # images_placeholder = tf.get_default_graph().get_tensor_by_name("input_tensor")
                embeddings = tf.get_default_graph().get_tensor_by_name("clip_by_value:0")



# read_write_new()
test_read()
# test_prdict()

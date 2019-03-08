import argparse
import pdb
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.graph_util import convert_variables_to_constants

import tensorflow as tf
from tensorflow.python.platform import gfile

from PIL import Image
import numpy as np
import os

input_file = "G:\\dataset\\Set5_bicubic\\bird.png"
output_dir = "G:\\dataset\\testwdsr"
pb_file = "wdsr_value.pb"

input_file = "/home/Cambricon-MLU100/datasets/Set5_bicubic/bird.png"
output_dir = "out_result"


def test_read():
    with tf.Session() as sess:
        with gfile.FastGFile(pb_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            # print(graph_def)
            tf.import_graph_def(graph_def, name='')
            input_tensor = sess.graph.get_tensor_by_name('input_tensor:0')
            output_tensor = tf.get_default_graph().get_tensor_by_name("clip_by_value:0")

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


if __name__ == '__main__':
    test_read()

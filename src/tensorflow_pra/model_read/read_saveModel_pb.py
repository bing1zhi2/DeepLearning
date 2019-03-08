import tensorflow as tf
import sys

import numpy as np
from PIL import Image
import os
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver as tf_saver
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

# OP_NOT_SUPPORT = ["Identity", "Conv2D", "Relu6","MaxPool",
#     "FusedBatchNorm", "ConcatV2", "AvgPool", "BiasAdd", "Shape"]

OP_NOT_SUPPORTED = ["BiasAdd",
                    "All",
                    "Assert",
                    "Cast",
                    "Fill",
                    "Gather",
                    "NonMaxSuppressionV2",
                    "Range",
                    "Rank",
                    "RealDiv",
                    "Size",
                    "Sqrt",
                    "TensorArrayV3",
                    "Tile",
                    "TopKV2",
                    "Where",
                    "ZerosLike"
                    ]

input_model_dir = 'F:\\dataset\\pre_train_model\\wdsr\\saved_model.pb'


def write():
    with tf.Session() as sess:
        model_filename = input_model_dir
        with gfile.FastGFile(model_filename, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)

            if 1 != len(sm.meta_graphs):
                print('More than one graph found. Not sure which to write')
                sys.exit(1)

            graphs = sm.meta_graphs
            # print(graphs)
            g_def = graphs[0].graph_def
            g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
            for node in g_def.node:
                # print("--------------------", node.name, ", ", node.op, ", ", node.device)
                if node.op in OP_NOT_SUPPORTED:
                    node.device = '/device:CPU:0'
            for node in g_def.node:
                print("--------------------", node.name, ", ", node.op, ", ", node.device)

        with tf.gfile.FastGFile('test.pb', "w") as f:
            f.write(g_def.SerializeToString())
            f.close()


#   g_in = tf.import_graph_def(graphs[0].graph_def)
# LOGDIR='./logdir'
# train_writer = tf.summary.FileWriter(LOGDIR)
# train_writer.add_graph(sess.graph)
# train_writer.flush()
# train_writer.close()

def read_wdsr():
    model_test = 'F:\\dataset\\pre_train_model\\wdsr_test'
    # with tf.Session(graph=tf.Graph()) as sess:
    #   metagraph_def = tf.saved_model.loader.load(
    #     sess, [tf.saved_model.tag_constants.SERVING], model_test)

    with tf.Session() as sess:
        model_filename = 'F:\\dataset\\pre_train_model\\wdsr\\saved_model.pb'
        export_dir = 'F:\\dataset\\pre_train_model\\wdsr\\saved_model.pb'
        with gfile.FastGFile(model_filename, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)

            if 1 != len(sm.meta_graphs):
                print('More than one graph found. Not sure which to write')
                sys.exit(1)

            graphs = sm.meta_graphs
            # print(graphs)
            g_def = graphs[0].graph_def
            # g_in = tf.import_graph_def(g_def)

            # Build a saver by importing the meta graph def to load.
            saver = tf_saver.import_meta_graph(graphs[0])

            if saver:
                # Build the checkpoint path where the variables are located.
                variables_path = os.path.join(
                    compat.as_bytes(export_dir),
                    compat.as_bytes("variables"),
                    compat.as_bytes("variables"))

                # Restore the variables using the built saver in the provided session.
                saver.restore(sess, variables_path)
            else:
                print("The specified SavedModel has no variables; no "
                                "checkpoints were restored.")



            input_tensor = sess.graph.get_tensor_by_name('import/input_tensor:0')
            output_tensor = sess.graph.get_tensor_by_name('import/clip_by_value:0')

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


# write()
read_wdsr()

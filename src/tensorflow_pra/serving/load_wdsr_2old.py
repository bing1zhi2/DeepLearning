from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

from tensorflow.python.framework import graph_util
from tensorflow.python.framework.graph_util import convert_variables_to_constants

model_dir = "F:\\dataset\\pre_train_model\\wdsr"
output_dir = 'temp/'
OP_NOT_SUPPORTED = ["BiasAdd"
                    ]


def save_newmodel_to_old():
    with tf.Session(graph=tf.Graph()) as sess:
        metagraph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], model_dir)
        signature_def = metagraph_def.signature_def[
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        input_tensor = sess.graph.get_tensor_by_name(
            signature_def.inputs['inputs'].name)

        output_tensor = sess.graph.get_tensor_by_name(
            signature_def.outputs['output'].name)

        print(signature_def.outputs['output'].name)

        predict_signature_def = metagraph_def.signature_def[
            tf.saved_model.signature_constants.PREDICT_METHOD_NAME]

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['clip_by_value'])


def predict():
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--model-dir', help='GCS location to load exported model', required=True)
    #
    # parser.add_argument(
    #     '--output-dir', help='GCS location to load output images', required=True)
    #
    # args = parser.parse_args()

    with tf.Session(graph=tf.Graph()) as sess:
        metagraph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], model_dir)
        signature_def = metagraph_def.signature_def[
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        input_tensor = sess.graph.get_tensor_by_name(
            signature_def.inputs['inputs'].name)
        output_tensor = sess.graph.get_tensor_by_name(
            signature_def.outputs['output'].name)

        predict_signature_def = metagraph_def.signature_def[
            tf.saved_model.signature_constants.PREDICT_METHOD_NAME]

        # if not os.path.isdir(args.output_dir):
        #   os.mkdir(args.output_dir)

        # print all tensors in checkpoint file
        # chkp.print_tensors_in_checkpoint_file(model_dir + '/variables/variables', tensor_name='', all_tensors=True,
        #                                       all_tensor_names=True)

        # print(metagraph_def)

        sss = output_dir + "mymodel"
        # if os.path.isdir(sss):
        #     os.removedirs(sss)

        # tensor_info_x = tf.saved_model.utils.build_tensor_info(input_tensor)
        # tensor_info_y = tf.saved_model.utils.build_tensor_info(output_tensor)

        # for node in metagraph_def.graph_def.node:
        #     if node.op in OP_NOT_SUPPORTED:
        #         node.device = '/device:CPU:0'
        # for node in metagraph_def.graph_def.node:
        #     if node.op in OP_NOT_SUPPORTED:
        #         print(node.device)

        # tf.saved_model.simple_save(sess,sss,
        #                            inputs={'inputs': input_tensor},
        #                            outputs={'output': output_tensor})

        # prediction_signature = (
        #     tf.saved_model.signature_def_utils.build_signature_def(
        #         inputs={'inputs': tensor_info_x},
        #         outputs={'output': tensor_info_y},
        #         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder = tf.saved_model.builder.SavedModelBuilder(sss)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature_def,
                tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
                    predict_signature_def,
            },
            main_op=None,
            strip_default_attrs=True)

    #     graph =sess.graph_def
    #     for node in graph.node:
    #         if node.op in OP_NOT_SUPPORTED:
    #             node.device = '/device:CPU:0'
    # # #
    # with tf.Session(graph=graph) as sess:
    #     builder.add_meta_graph([tf.saved_model.tag_constants.TPU],
    #                            signature_def_map={
    #                                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
    #                                    signature_def,
    #                                tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
    #                                    predict_signature_def,
    #                            },
    #                            strip_default_attrs=True)

    builder.save()


def test_var():
    # chkp.print_tensors_in_checkpoint_file('temp/mymodel/variables/variables', tensor_name='', all_tensors=True,
    #                                       all_tensor_names=True)
    output_dir = "G:\\dataset\\testwdsr"
    with tf.Session(graph=tf.Graph()) as sess:
        metagraph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], model_dir)
        print(metagraph_def)

        signature_def = metagraph_def.signature_def[
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        input_tensor = sess.graph.get_tensor_by_name(
            signature_def.inputs['inputs'].name)
        output_tensor = sess.graph.get_tensor_by_name(
            signature_def.outputs['output'].name)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        input_file = "G:\\dataset\\Set5_bicubic\\bird.png"
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


# predict()
# test_var()
save_newmodel_to_old()

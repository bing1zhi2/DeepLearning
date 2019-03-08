from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from PIL import Image
import numpy as np

import tensorflow as tf


def predict():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model-dir', help='GCS location to load exported model', required=True)
  parser.add_argument(
      '--input_file', help='GCS location to load input images', required=True)
  parser.add_argument(
      '--output-dir', help='GCS location to load output images', required=True)
  parser.add_argument(
      '--ensemble',
      help='Whether to ensemble with 8x rotation and flip',
      default=False,
      action='store_true')
  args = parser.parse_args()

  with tf.Session(graph=tf.Graph()) as sess:
    metagraph_def = tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], args.model_dir)
    print(metagraph_def)

    signature_def = metagraph_def.signature_def[
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_tensor = sess.graph.get_tensor_by_name(
        signature_def.inputs['inputs'].name)
    output_tensor = sess.graph.get_tensor_by_name(
        signature_def.outputs['output'].name)
    if not os.path.isdir(args.output_dir):
      os.mkdir(args.output_dir)

    input_file = args.input_file
    output_file = os.path.join(args.output_dir, 'out.png')
    input_image = np.asarray(Image.open(input_file))

    def forward_images(images):
        images = images.astype(np.float32) / 255.0
        with tf.device("/device:CPU:0"):
            images = output_tensor.eval(feed_dict={input_tensor: images})
        return images

    input_images = np.expand_dims(input_image, axis=0)
    output_images = forward_images(input_images)
    output_image = output_images[0]

    output_image = np.around(output_image * 255.0).astype(np.uint8)
    output_image = Image.fromarray(output_image, 'RGB')
    output_image.save(output_file)


predict()
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from PIL import Image
import numpy as np

import tensorflow as tf

model_dir="model/1"
input_file="test.jpg"

def preprocess_image(im):
    #等比例将图像高度缩放到32
    # im=tf.image.rgb_to_grayscale(im)
    im_shape = tf.shape(im)
    h=im_shape[1]
    w=im_shape[2]
    height=tf.constant(32,tf.int32)
    scale = tf.divide(tf.cast(h,tf.float32),32)
    width = tf.divide(tf.cast(w,tf.float32),scale)
    width =tf.cast(width,tf.int32)
    resize_image = tf.image.resize_images(im, [height,width], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    resize_image = tf.cast(resize_image, tf.float32) / 255 - 0.5
    width = tf.reshape(width, [1])
    height = tf.reshape(height, [1])
    im_info = tf.concat([height, width], 0)
    im_info = tf.concat([im_info, [1]], 0)
    im_info = tf.reshape(im_info, [1, 3])
    im_info = tf.cast(im_info, tf.float32)
    return resize_image,im_info

with tf.Session(graph=tf.Graph()) as sess:
    metagraph_def = tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], model_dir)
    # print(metagraph_def)

    signature_def = metagraph_def.signature_def[
        "predict_images"]
    input_tensor = sess.graph.get_tensor_by_name(
        signature_def.inputs['images'].name)
    output_tensor = sess.graph.get_tensor_by_name(
        signature_def.outputs['prediction'].name)

    input_image = np.asarray(Image.open(input_file))


    def forward_images(images):
        with tf.device("/device:CPU:0"):
            images = output_tensor.eval(feed_dict={input_tensor: images})
        return images

    img = np.asarray(Image.open(input_file))
    #
    img = Image.fromarray(img).convert('L')


    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)

    img = img.resize([width, 32], Image.ANTIALIAS)

    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    img = np.array(img).astype(np.float32) / 255.0 - 0.5

    X = img.reshape([1, 32, width, 1])

    y = forward_images(X)
    print(y)





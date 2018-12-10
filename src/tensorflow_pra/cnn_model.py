# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from tensorflow.python.framework import graph_util

input = tf.Variable(tf.constant(1.0, shape=[1, 5, 5, 1]), name='test')

filter1 = tf.Variable(tf.constant([-1.0, 0, 0, -1], shape=[2, 2, 1, 1]))

op1 = tf.nn.conv2d(input, filter1, strides=[1, 2, 2, 1], padding='SAME', name='op1')

init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)

    print("op1:\n", sess.run([op1, filter1]))  # 1-1  后面补0
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op1'])
    print(constant_graph)
    saver.save(sess, 'checkpoint/model')
    with tf.gfile.GFile('checkpoint/model.pb', 'wb') as f:
        f.write(constant_graph.SerializeToString())
    print("%d ops in the final graph: %s" % (len(constant_graph.node), 'checkpoint/model.pb'))

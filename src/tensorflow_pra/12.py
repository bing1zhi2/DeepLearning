#!/usr/bin/python
# -*- coding: UTF-8 -*-
from tensorflow_pra.examples.tutorials.mnist import input_data
import tensorflow_pra as tf

my_mnist = input_data.read_data_sets("MNIST_data_bak/", one_hot=True)

# 784表示每张图片都是一个784个像素值的一维向量
x = tf.placeholder(dtype=tf.float32, shape=(None, 784))
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(dtype=tf.float32, shape=(None, 10))
# 定义损失函数，交叉熵损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = my_mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print("TrainSet batch acc : %s  " % accuracy.eval({x: batch_xs, y_: batch_ys}))
    print("ValidSet acc : %s" % accuracy.eval({x: my_mnist.validation.images, y_: my_mnist.validation.labels}))

# 测试
print("TestSet acc : %s" % accuracy.eval({x: my_mnist.test.images, y_: my_mnist.test.labels}))

# -*- coding:UTF-8 -*-
import numpy as np
import tensorflow as tf

# 基本常量
x = tf.constant([[1.0, 2.0]])
w = tf.constant([[3.0], [4.0]])
y = tf.matmul(x, w)
print(y)
with tf.Session() as sess:
    print(sess.run(y))

## reduce_sum 用法，keepdims保持维度
x = tf.constant([[1, 1, 1], [1, 1, 1]])
tf.reduce_sum(x)  # 6
tf.reduce_sum(x, 0)  # [2, 2, 2]
tf.reduce_sum(x, 1)  # [3, 3]
tf.reduce_sum(x, reduction_indices=[1])  # [3, 3] reduction_indices 为axis的过期版本
tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
tf.reduce_sum(x, [0, 1])  # 6

x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1.0, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1.0, seed=1))
c1 = tf.matmul(x, w1)
y = tf.matmul(c1, w2)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    result = sess.run(y, feed_dict={x: [[1.0, 2.0], [0.1, 0.3]]})
    print('y:', result)
rand = np.random.RandomState(333)
X = rand.rand(32, 2)

# 高，宽，通道，卷积核
filters_test = np.zeros(shape=(3, 3, 1, 2), dtype=np.float32)
print(filters_test)
print("--------------------")
filters_test[:, 2, :, 0] = 1
print(filters_test)
print("--------------------")
filters_test[2, :, :, 1] = 1
print(filters_test)
print("--------------------")


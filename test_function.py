import numpy as np
import tensorflow as tf

x =tf.constant([[1.0,2.0]])
w =tf.constant([[3.0],[4.0]])
y= tf.matmul(x,w)
print(y)
with tf.Session() as sess:
    print(sess.run(y))

## reduce_sum
x = tf.constant([[1, 1, 1], [1, 1, 1]])
tf.reduce_sum(x)  # 6
tf.reduce_sum(x, 0)  # [2, 2, 2]
tf.reduce_sum(x, 1)  # [3, 3]
tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
tf.reduce_sum(x, [0, 1])  # 6

rand= np.random.RandomState(333)
X= rand.rand(32,2)



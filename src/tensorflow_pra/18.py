import numpy as np
from sklearn.datasets import load_sample_images
import tensorflow_pra as tf
import matplotlib.pyplot as plt

dataset = np.array(load_sample_images().images, dtype=np.float32)
# mini-batch通常是4D，[mini-batch size, height, width, channels]
print(dataset.shape)  # (2, 427, 640, 3)
batch_size, height, width, channels = dataset.shape

# 高，宽，通道，卷积核
filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
print(filters_test)
print("--------------------")
filters_test[:, 3, :, 0] = 1
print(filters_test)
print("--------------------")
filters_test[3, :, :, 1] = 1
print(filters_test)
print("--------------------")

X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filter=filters_test, strides=[1, 2, 2, 1], padding='SAME')
with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})

plt.imshow(output[0, :, :, 0])  # 绘制第一个图的第一个特征图
plt.show()

plt.imshow(output[0, :, :, 1])  # 绘制第一个图的第二个特征图
plt.show()

plt.imshow(output[1, :, :, 0])  # 绘制第二个图的第一个特征图
plt.show()

plt.imshow(output[1, :, :, 1])  # 绘制第二个图的第二个特征图
plt.show()
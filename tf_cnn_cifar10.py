import tensorflow_pra as tf
import numpy as np
import math
import time
from tutorials.image.cifar10 import cifar10
from tutorials.image.cifar10 import cifar10_input

max_steps = 100
batch_size = 128
# 下载cifar10数据集的默认路径
data_dir = 'G:/ML/dataset/cifar10_data/cifar-10-batches-bin'

# 下载cifar10类下载数据集，并解压，展开到其默认位置
cifar10.maybe_download_and_extract()
images_train, labels_train = cifar10_input.distorted_inputs(
    data_dir=data_dir, batch_size=batch_size
)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

# 样本数 ，24*24图片裁剪后大小，3个颜色通道
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])


def variable_with_weight_losses(shape, stddev, wl):
    # 定义初始化weights的函数，和之前一样依然使用tf.truncated_normal截断的正太分布来初始化权值
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        # 给weight加一个L2的loss，相当于做了一个L2的正则化处理
        # 在机器学习中，不管是分类还是回归任务，都可能因为特征过多而导致过拟合，一般可以通过减少特征或者惩罚不重要特征的权重来缓解这个问题
        # 但是通常我们并不知道该惩罚哪些特征的权重，而正则化就是帮助我们惩罚特征权重的，即特征的权重也会成为模型的损失函数的一部分
        # 我们使用w1来控制L2 loss的大小
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        # 我们使用tf.add_to_collection把weight loss统一存到一个collection，这个collection名为"losses"，它会在后面计算神经网络
        # 总体loss时被用上
        tf.add_to_collection("losses", weight_loss)
    return var


weight1 = variable_with_weight_losses(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
kernel_1 = tf.nn.conv2d(image_holder, filter=weight1, strides=[1, 1, 1, 1], padding='SAME')
bias_1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv_1 = tf.nn.relu(tf.nn.bias_add(kernel_1, bias_1))
pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norml = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 第二个卷积层
# 上面64个卷积核，即输出64个通道，所以本层卷积核尺寸的第三个维度即输入的通道数也需要调整为64
weight2 = variable_with_weight_losses(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel_2 = tf.nn.conv2d(norml, weight2, [1, 1, 1, 1], padding='SAME')
bias_2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv_2 = tf.nn.relu(tf.nn.bias_add(kernel_2, bias_2))
norm_2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

reshape = tf.reshape(pool_2, [batch_size, -1])
dim = reshape.get_shape()[1].value

weight3 = variable_with_weight_losses(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
# 最后我们依然使用ReLU激活函数进行非线性化
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

weight4 = variable_with_weight_losses(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

weight5 = variable_with_weight_losses(shape=[192, 10], stddev=1 / 192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


loss = loss(logits=logits, labels=label_holder)
# 优化器依然选择Adam Optimizer, 学习速率0.001
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 使用 tf.nn.in_top_k()函数求输出结果中 top k的准确率，默认使用top 1，也就是输出分数最高的那一类的准确率
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 前面对图像进行数据增强的操作需要耗费大量CPU时间，因此distorted_inputs使用了16个独立的线程来加速任务，函数内部会产生线程池，
# 在需要使用时会通过TensorFlow queue进行调度
# 启动图片数据增强的线程队列，这里一共使用了16个线程来进行加速，如果不启动线程，那么后续inference以及训练的操作都是无法开始的
tf.train.start_queue_runners()

# 进行训练
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss],
                             feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = 'step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

num_examples = 10000
# 先计算一共要多少个batch才能将全部样本评测完
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                  label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

my_mnist = input_data.read_data_sets("G:\ML\dataset\MNIST_data_bak", one_hot=True)
# each image is 28*28
# 构造一个简单的网络来实现手写数字识别
x = tf.placeholder(dtype=tf.float32, shape=(None, 784))
w = tf.Variable(tf.zeros(shape=(784, 10)))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 10))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), 1))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

n_epoch = 1000

with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(n_epoch):
        batch_xs, batch_ys = my_mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if epoch % 100 == 0:
            cross_entropy_v = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
            print('after epoch %d the loss is %f' % (epoch, cross_entropy_v))
            save_path = saver.save(sess, "./ckpt/my_model.ckpt")
    saver.save(sess, "./ckpt/my_model_final.ckpt")
    # 恢复模型
    saver.restore(sess, "./ckpt/my_model_final.ckpt")
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({x: my_mnist.test.images, y_: my_mnist.test.labels}))

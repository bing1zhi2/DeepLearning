import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.contrib.layers import fully_connected


# 构建图阶段
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2,scope="hidden2")
    # 最后一层不使用激活函数，因为下面要损失函数
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)

with tf.name_scope("loss"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name="loss")

learn_rate=0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    traing_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

mnist = input_data.read_data_sets("G:\ML\dataset\MNIST_data_bak")
n_epochs = 400
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iter in range(mnist.train.num_examples // batch_size):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(traing_op, feed_dict={X: x_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: x_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_dnn_model_final.ckpt")

'''
# 使用模型预测
with tf.Session as sess:
    saver.restore(sess, "./my_dnn_model_final.ckpt")
    X_new_scaled = [...]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)  # 查看最大的类别是哪个
'''
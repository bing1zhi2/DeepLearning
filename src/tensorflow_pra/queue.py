# -*- coding:utf-8 -*-
'''
队列  多线程输入
'''
import tensorflow as tf

# 创建长为100的队列
queue = tf.FIFOQueue(100, "float")
c = tf.Variable(0.0) # 计数器
op = tf.assign_add(c, tf.constant(1.0)) # 加1
enqueue_op = queue.enqueue(c) # 入队列
qr= tf.train.QueueRunner(queue, enqueue_ops=[op, enqueue_op])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator() # 协调器
    # 启动入队线程
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

    #主线程
    for i in range(0, 10):
        print("--------------")
        print(sess.run(queue.dequeue()))

    coord.request_stop() # 通知其他线程关闭，其他关闭后它才返回
    # coord.join(enqueue_threads) # 等待其他线程


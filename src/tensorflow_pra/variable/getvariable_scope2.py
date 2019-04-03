# -*- coding:utf-8 -*-
import tensorflow as tf

tf.reset_default_graph()

# var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
# var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)

with tf.variable_scope("test1", ):
    var1 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

    with tf.variable_scope("test2"):
        var2 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

print("var1:", var1.name)
print("var2:", var2.name)

'''
var1: test1/firstvar:0
var2: test1/test2/firstvar:0
'''

# reuse=True  不新建变量，使用图中变量
with tf.variable_scope("test1", reuse=True):
    var3 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)
    with tf.variable_scope("test2"):
        var4 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)

print("var3:", var3.name)
print("var4:", var4.name)

'''
var3: test1/firstvar:0
var4: test1/test2/firstvar:0
'''

'''
可以 把 var1 和 var2 放到 一个 网络 模型 里 去 训练，
 把 var3 和 var4 放到 另一个 网络 模型 里 去 训练， 
 而 两个 模型 的 训练 结果 都会 作用于 一个 模型 的 学习 参数 上。


'''
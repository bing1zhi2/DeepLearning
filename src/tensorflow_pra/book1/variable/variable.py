# -*- coding:utf-8 -*-
"""
共享变量
"""
import tensorflow as tf

tf.reset_default_graph()

var1 = tf.Variable(1.0, name="var1")
print("var1", var1.name)
var1 = tf.Variable(2.0, name="var1")
print("var1", var1.name)

'''
var1 var1:0
var1 var1_1:0
'''

var2 = tf.Variable(3.0)
print("var2", var2.name)
var2 = tf.Variable(4.0)
print("var2", var2.name)
'''
var2 Variable:0
var2 Variable_1:0
'''

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("var1=", var1.eval())
    print("var2=", var2.eval())

'''
var1= 2.0
var2= 4.0
'''


get_var1 = tf.get_variable("var1",[1],initializer=tf.constant_initializer(0.3))
print("get var1", get_var1.name)

# 不能使用 get_variable 定义 同名
# get_var1 = tf.get_variable("var1",[1], initializer=tf.constant_initializer(0.4))
# print ("get_var1:",get_var1.name)

get_var1 = tf.get_variable("var11",[1], initializer=tf.constant_initializer(0.4))
print ("get_var1:",get_var1.name)

'''
get var1 var1_2:0
get_var1: var11:0
'''

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("get_var1=",get_var1.eval())

'''
get_var1= [0.4]
'''
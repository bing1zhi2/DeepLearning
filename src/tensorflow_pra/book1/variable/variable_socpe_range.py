# -*- coding:utf-8 -*-
"""
 变量作用域的范围   as 使用
"""

import tensorflow as tf

tf.reset_default_graph()

with tf.variable_scope("scope1") as sp:
    var1 = tf.get_variable("v", [1])

print("sp:", sp.name)
print("var1:", var1.name)
"""
sp: scope1
var1: scope1/v:0

"""

with tf.variable_scope("scope2"):
    var2 = tf.get_variable("v", [1])

    with tf.variable_scope(sp) as sp1:
        var3 = tf.get_variable("v3", [1])

        with tf.variable_scope(""):
            var4 = tf.get_variable("v4", [1])

print("sp1:", sp1.name)
print("var2:", var2.name)
print("var3:", var3.name)
print("var4:", var4.name)

"""

sp1: scope1
var2: scope2/v:0
var3: scope1/v3:0   var3 的范围仍在scope1 下不受上层影响
var4: scope1//v4:0   var4多了一个空层
"""


with tf.variable_scope("scope"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        x = 1.0 + v
        with tf.name_scope(""):
            y = 1.0 + v
print("v:",v.name)
print("x.op:",x.op.name)
print("y.op:",y.op.name)

"""
v: scope/v:0
x.op: scope/bar/add
y.op: add   name_scope 只限制op，使name_scope用空白时，返回顶层，不是多空层
"""
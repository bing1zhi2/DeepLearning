# 导入TensorFlow和tf.keras
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
# 导入辅助库
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("ddd")
x_train, x_test = x_train / 255.0, x_test / 255.0


# fashion_mnist = keras.datasets.fashion_mnist
#
#
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#
print(x_train.shape)
print(len(x_train))


plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show()



class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i]])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)

])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])


model.summary()

model.fit(x_train, y_train, epochs=9)

test_loss, test_acc = model.evaluate(x_test, y_test)

print(test_loss, test_acc)


# 从测试数据集中获取图像
img = x_test[0]

print(img.shape)

# 将图像添加到批次中，即使它是唯一的成员。
img = (np.expand_dims(img,0))

print(img.shape)


predictions_single = model.predict(img)

print(predictions_single)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

plot_value_array(0, predictions_single, y_test)
plt.xticks(range(10), class_names, rotation=45)
plt.show()
# 导入TensorFlow和tf.keras
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
# 导入辅助库
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("ddd")
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train[0].shape)

X_train = x_train.reshape(-1, 28, 28, 1)
X_test = x_test.reshape(-1, 28, 28, 1)

# train_datagen = ImageDataGenerator(rescale=1.0 / 255)

# train_generator = train_datagen.flow(x_train,y_train,batch_size=100)



inputs = Input((28, 28 , 1))
x = inputs

x = Conv2D(32, (5, 5), padding='same', activation='relu')(x)
x = MaxPooling2D(2)(x)
x = Conv2D(64, (5,5), padding='same', activation='relu')(x)
x = MaxPooling2D(2)(x)
x = Flatten()(x)
x = Dense(100, activation='sigmoid')(x)
x = Dense(10, activation='sigmoid')(x)

model = Model(inputs, x)


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])


model.summary()

h = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# test_loss, test_acc = model.evaluate(X_test, y_test)
#
# print(test_loss, test_acc)

model.save("minist_cnn.h5")



plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')

plt.subplot(1, 2, 2)
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.ylabel('acc')
plt.xlabel('epoch')

plt.show()


# 从测试数据集中获取图像
img = X_test[0]

plt.figure()
plt.imshow(x_test[0])
plt.show()

print(img.shape)
# 将图像添加到批次中，即使它是唯一的成员。
img = (np.expand_dims(img,0))

print(img.shape)



predictions_single = model.predict(img)

print(predictions_single)

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)
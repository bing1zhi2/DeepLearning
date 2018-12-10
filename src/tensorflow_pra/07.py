import tensorflow_pra as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(data_home="G:\ML\dataset\scikit_learn_data",download_if_missing=True)
m, n= housing.data.shape
print(m, n)
print(housing.target.shape)
print(housing.target)
#print(housing.data, housing.target)
print(housing.feature_names)
housing_data_plus_bias= np.c_[np.ones((m, 1)), housing.data]
y_pre= housing.target.reshape(-1, 1)
print(y_pre.shape)

X= tf.constant(housing_data_plus_bias, dtype=tf.float32, name ='X')
Y = tf.constant(y_pre,dtype=tf.float32, name='y')
XT = tf.transpose(X)

theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), Y)
with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)

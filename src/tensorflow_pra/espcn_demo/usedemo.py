import tensorflow as tf
from scipy import ndimage
from scipy import misc
import numpy as np
from tensorflow_pra.espcn_demo.espcn import ESPCN
from tensorflow_pra.espcn_demo.prepare_data import *

'''
params['filters_size'],
                   channels=params['channels'],
                   ratio=params['ratio'],
                   batch_size=1,
                   lr_size=params['lr_size'],
                   edge=params['edge']
'''


def super_resolution():
    '''

    :return:
    '''
    filters_size = [5, 3, 3]
    channels = [64, 32]

    batch_size = 1
    lr_size = 17
    edge = 8

    ratio = 2
    checkpoint='logdir_2x/train'
    # ratio = 3
    # checkpoint = 'logdir_3x/train'

    lr_image_dir='F:/code/other/ml/ESPCN-TensorFlow/images/butterfly_GT.jpg'
    out_path='butterfly_HR_2x'
    g1 = tf.Graph()
    with tf.Session(graph= g1) as sess:
        net = ESPCN(filters_size=filters_size,
                    channels=channels,
                    ratio=ratio,
                    batch_size=batch_size,
                    lr_size=lr_size, edge=edge)


        lr_image_data = misc.imread(lr_image_dir)
        lr_image_ycbcr_data = rgb2ycbcr(lr_image_data)
        lr_image_y_data = lr_image_ycbcr_data[:, :, 0:1]
        lr_image_cb_data = lr_image_ycbcr_data[:, :, 1:2]
        lr_image_cr_data = lr_image_ycbcr_data[:, :, 2:3]
        lr_image_batch = np.zeros((1,) + lr_image_y_data.shape)
        lr_image_batch[0] = lr_image_y_data

        lr_image = tf.placeholder(tf.uint8)
        sr_image = net.generate(lr_image)

        saver = tf.train.Saver()
        try:
            model_loaded = net.load(sess, saver, checkpoint)
        except:
            raise Exception(
                "Failed to load model, does the ratio in params.json match the ratio you trained your checkpoint with?")

        if model_loaded:
            print("[*] Checkpoint load success!")
        else:
            print("[*] Checkpoint load failed/no checkpoint found")
            return
        print(lr_image.name, sr_image.name)

        tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        tf.get_default_graph().get_tensor_by_name("Cast_2:0")


        sr_image_y_data = sess.run(sr_image, feed_dict={lr_image: lr_image_batch})

        sr_image_y_data = shuffle(sr_image_y_data[0], ratio)
        sr_image_ycbcr_data = misc.imresize(lr_image_ycbcr_data,
                                            ratio * np.array(lr_image_data.shape[0:2]),
                                            'bicubic')

        edge = int(edge * ratio / 2)

        sr_image_ycbcr_data = np.concatenate((sr_image_y_data, sr_image_ycbcr_data[edge:-edge, edge:-edge, 1:3]),
                                             axis=2)
        sr_image_data = ycbcr2rgb(sr_image_ycbcr_data)

        misc.imsave(out_path + '.png', sr_image_data)

        # LOGDIR = './logdir'
        # train_writer = tf.summary.FileWriter(LOGDIR)
        # train_writer.add_graph(sess.graph)
        # train_writer.flush()
        # train_writer.close()


def super_resolution2():
    '''

    :return:
    '''
    filters_size = [5, 3, 3]
    channels = [64, 32]

    batch_size = 1
    lr_size = 17
    edge = 8


    ratio = 3
    checkpoint = 'logdir_3x/train'

    lr_image_dir='F:/code/other/ml/ESPCN-TensorFlow/images/butterfly_GT.jpg'
    out_path='butterfly_HR_3x'

    g2 = tf.Graph()
    with tf.Session(graph=g2) as sess:
        net = ESPCN(filters_size=filters_size,
                    channels=channels,
                    ratio=ratio,
                    batch_size=batch_size,
                    lr_size=lr_size, edge=edge)


        lr_image_data = misc.imread(lr_image_dir)
        lr_image_ycbcr_data = rgb2ycbcr(lr_image_data)
        lr_image_y_data = lr_image_ycbcr_data[:, :, 0:1]
        lr_image_cb_data = lr_image_ycbcr_data[:, :, 1:2]
        lr_image_cr_data = lr_image_ycbcr_data[:, :, 2:3]
        lr_image_batch = np.zeros((1,) + lr_image_y_data.shape)
        lr_image_batch[0] = lr_image_y_data

        lr_image = tf.placeholder(tf.uint8)
        sr_image = net.generate(lr_image)

        saver = tf.train.Saver()
        try:
            model_loaded = net.load(sess, saver, checkpoint)
        except:
            raise Exception(
                "Failed to load model, does the ratio in params.json match the ratio you trained your checkpoint with?")

        if model_loaded:
            print("[*] Checkpoint load success!")
        else:
            print("[*] Checkpoint load failed/no checkpoint found")
            return
        print(lr_image.name, sr_image.name)

        tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        tf.get_default_graph().get_tensor_by_name("Cast_2:0")


        sr_image_y_data = sess.run(sr_image, feed_dict={lr_image: lr_image_batch})

        sr_image_y_data = shuffle(sr_image_y_data[0], ratio)
        sr_image_ycbcr_data = misc.imresize(lr_image_ycbcr_data,
                                            ratio * np.array(lr_image_data.shape[0:2]),
                                            'bicubic')

        edge = int(edge * ratio / 2)

        sr_image_ycbcr_data = np.concatenate((sr_image_y_data, sr_image_ycbcr_data[edge:-edge, edge:-edge, 1:3]),
                                             axis=2)
        sr_image_data = ycbcr2rgb(sr_image_ycbcr_data)

        misc.imsave(out_path + '.png', sr_image_data)

        # LOGDIR = './logdir'
        # train_writer = tf.summary.FileWriter(LOGDIR)
        # train_writer.add_graph(sess.graph)
        # train_writer.flush()
        # train_writer.close()



super_resolution()
super_resolution2()
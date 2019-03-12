# -*- coding:utf-8 -*-
import tensorflow as tf
import os

from scipy import misc
import numpy as np
from tensorflow_pra.espcn_demo.espcn import ESPCN
from tensorflow_pra.espcn_demo.prepare_data import *
from tensorflow_pra.espcn_demo import model_loader
from tensorflow_pra.espcn_demo import predict_param as param


# filters_size = [5, 3, 3]
# channels = [64, 32]
#
# batch_size = 1
# lr_size = 17
# edge = 8


# checkpoint = 'logdir_2x/train'
#
# checkpoint = 'logdir_3x/train'


# def loader_d():
#     with tf.Session() as sess:
#         # meta_file, ckpt_file = model_loader.get_model_filenames(checkpoint)
#         # saver = tf.train.import_meta_graph(os.path.join(checkpoint, meta_file))
#         # saver.restore(tf.get_default_session(), os.path.join(checkpoint, ckpt_file))
#
#         print("[*] Reading checkpoints...")
#         ckpt = tf.train.get_checkpoint_state(checkpoint)
#         saver = tf.train.Saver()
#         if ckpt and ckpt.model_checkpoint_path:
#             ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
#             saver.restore(sess, os.path.join(checkpoint, ckpt_name))
#             return True
#         else:
#             return False
#
#         tf.get_default_graph().get_tensor_by_name("Placeholder:0")
#         tf.get_default_graph().get_tensor_by_name("Cast_2:0")


class Predict:
    def __init__(self, model_exp, ratio):
        self.sr_espcn = SR_ESPCN(ratio, model_exp)


class SR_ESPCN:
    def __init__(self, ratio, checkpoint):
        self.ratio = ratio
        self.run_fun = self._setup_espcn(ratio, checkpoint)

    def _setup_espcn(self, ratio, checkpoint):
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                return self.espcn(sess, ratio, checkpoint)

    def espcn(self, sess, ratio, checkpoint):
        net = ESPCN(filters_size=param.filters_size,
                    channels=param.channels,
                    ratio=ratio,
                    batch_size=param.batch_size,
                    lr_size=param.lr_size, edge=param.edge)

        lr_image = tf.placeholder(tf.uint8)
        sr_image = net.generate(lr_image)

        saver = tf.train.Saver()

        model_loaded = net.load(sess, saver, checkpoint)

        run_fun = lambda img: sess.run(sr_image, feed_dict={lr_image: img})
        return run_fun

    def deal_data(self, lr_image_path):
        lr_image_data = misc.imread(lr_image_path)
        lr_image_ycbcr_data = rgb2ycbcr(lr_image_data)
        lr_image_y_data = lr_image_ycbcr_data[:, :, 0:1]

        lr_image_batch = np.zeros((1,) + lr_image_y_data.shape)
        lr_image_batch[0] = lr_image_y_data

        return lr_image_batch

    def get_result(self, lr_image_dir):
        # lr_image_batch = self.deal_data(lr_image_dir)

        lr_image_data = misc.imread(lr_image_dir)
        lr_image_ycbcr_data = rgb2ycbcr(lr_image_data)
        lr_image_y_data = lr_image_ycbcr_data[:, :, 0:1]

        lr_image_batch = np.zeros((1,) + lr_image_y_data.shape)
        lr_image_batch[0] = lr_image_y_data

        sr_image_y_data = self.run_fun(lr_image_batch)

        sr_image_y_data = shuffle(sr_image_y_data[0], self.ratio)
        sr_image_ycbcr_data = misc.imresize(lr_image_ycbcr_data,
                                            self.ratio * np.array(lr_image_data.shape[0:2]),
                                            'bicubic')

        edge = int(param.edge * self.ratio / 2)

        sr_image_ycbcr_data = np.concatenate((sr_image_y_data, sr_image_ycbcr_data[edge:-edge, edge:-edge, 1:3]),
                                             axis=2)
        sr_image_data = ycbcr2rgb(sr_image_ycbcr_data)

        print(sr_image_data)


# ratio_p = 2
# checkpoint_ = 'logdir_2x/train'
ratio_p = 3
checkpoint_ = 'logdir_3x/train'

lr_image_dir1 = 'F:/code/other/ml/ESPCN-TensorFlow/images/butterfly_GT.jpg'
out_path = 'butterfly_HR_3x'

# pre = Predict(checkpoint_, ratio_p)
# pre.sr_espcn.get_result(lr_image_dir1)
sr_espcn1 = SR_ESPCN(ratio_p, checkpoint_)
sr_espcn1.get_result(lr_image_dir1)

print("--------------------------------------------------------------------")
# pre2 = Predict('logdir_2x/train', 2)
# pre2.sr_espcn.get_result(lr_image_dir1)
# loader_d()

sr_espcn1 = SR_ESPCN(2, 'logdir_2x/train', )
sr_espcn1.get_result(lr_image_dir1)

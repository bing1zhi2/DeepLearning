# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.platform import gfile

from tensorflow_pra.espcn_demo.espcn import ESPCN
from tensorflow_pra.espcn_demo.prepare_data import *




def freeze():
    filters_size = [5, 3, 3]
    channels = [64, 32]

    batch_size = 1
    lr_size = 17
    edge = 8

    ratio = 2
    checkpoint = 'logdir_2x/train'
    ratio = 3
    checkpoint = 'logdir_3x/train'

    lr_image_dir = 'F:/code/other/ml/ESPCN-TensorFlow/images/butterfly_GT.jpg'
    out_path = 'butterfly_HR_3x'

    with tf.Session() as sess:
        net = ESPCN(filters_size=filters_size,
                    channels=channels,
                    ratio=ratio,
                    batch_size=batch_size,
                    lr_size=lr_size, edge=edge)

        lr_image = tf.placeholder(tf.uint8)
        lr_image_data = misc.imread(lr_image_dir)
        lr_image_ycbcr_data = rgb2ycbcr(lr_image_data)
        lr_image_y_data = lr_image_ycbcr_data[:, :, 0:1]
        lr_image_cb_data = lr_image_ycbcr_data[:, :, 1:2]
        lr_image_cr_data = lr_image_ycbcr_data[:, :, 2:3]
        lr_image_batch = np.zeros((1,) + lr_image_y_data.shape)
        lr_image_batch[0] = lr_image_y_data

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

        graph_def = tf.get_default_graph().as_graph_def()

        GraphDef = convert_variables_to_constants(sess, graph_def, ['Cast_2'])

        with tf.gfile.FastGFile('espcn_value3x.pb', "w") as f:
            f.write(GraphDef.SerializeToString())
            f.close()


def test_read():
    with tf.Session() as sess:
        with gfile.FastGFile("espcn_value3x.pb","rb") as f:
            grah_def = tf.GraphDef()
            grah_def.ParseFromString(f.read())
            print(grah_def)
            tf.import_graph_def(grah_def)

            input_tensor = sess.graph.get_tensor_by_name('Placeholder:0')
            out_tensor = tf.get_default_graph().get_tensor_by_name("Cast_2:0")





# freeze()
test_read()

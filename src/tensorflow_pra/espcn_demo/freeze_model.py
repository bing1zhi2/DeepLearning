# -*- coding:utf-8 -*-
import time
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

        print(lr_image.name)
        print(sr_image.name)

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

        t_variables = tf.trainable_variables()
        print(t_variables)

        graph_def = tf.get_default_graph().as_graph_def()

        print(graph_def)

        GraphDef = convert_variables_to_constants(sess, graph_def, ['Cast_2'])
        print(GraphDef)

        with tf.gfile.FastGFile('espcn_value3x.pb', "w") as f:
            f.write(GraphDef.SerializeToString())
            f.close()


def test_read():
    model_pb = "espcn_value3x.pb"
    lr_image_dir = 'F:/code/other/ml/ESPCN-TensorFlow/images/butterfly_GT.jpg'
    out_path = "test_pb_butterfly"
    ratio = 3
    p_edge = 8
    with tf.Session() as sess:
        with gfile.FastGFile(model_pb,"rb") as f:
            grah_def = tf.GraphDef()
            grah_def.ParseFromString(f.read())
            tf.import_graph_def(grah_def)
            # print(tf.get_default_graph().as_graph_def())

            input_tensor = sess.graph.get_tensor_by_name('import/Placeholder:0')
            out_tensor = tf.get_default_graph().get_tensor_by_name("import/Cast_2:0")



            lr_image_data = misc.imread(lr_image_dir)
            lr_image_ycbcr_data = rgb2ycbcr(lr_image_data)
            lr_image_y_data = lr_image_ycbcr_data[:, :, 0:1]

            lr_image_batch = np.zeros((1,) + lr_image_y_data.shape)
            lr_image_batch[0] = lr_image_y_data

            time1 = time.time()

            sr_image_y_data = out_tensor.eval(feed_dict={input_tensor:lr_image_batch})
            print(sr_image_y_data)
            time2 = time.time()
            print("sess run success.................",time2 - time1)

            sr_image_y_data = shuffle(sr_image_y_data[0], ratio)
            sr_image_ycbcr_data = misc.imresize(lr_image_ycbcr_data,
                                                ratio * np.array(lr_image_data.shape[0:2]),
                                                'bicubic')

            edge = int(p_edge * ratio / 2)

            sr_image_ycbcr_data = np.concatenate((sr_image_y_data, sr_image_ycbcr_data[edge:-edge, edge:-edge, 1:3]),
                                                 axis=2)
            sr_image_data = ycbcr2rgb(sr_image_ycbcr_data)

            time3 = time.time()
            print('image trans...',time3-time2)

            misc.imsave(out_path + '.png', sr_image_data)





# freeze()
test_read()

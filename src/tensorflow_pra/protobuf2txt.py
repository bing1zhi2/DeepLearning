# coding=utf-8
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile
import sys
import argparse


# pb_path='/media/chenhao/study/dataset/facenet/model/facenet_model_20180402-114759/20180402-114759/20180402-114759.pb'
# pb_path='/home/chenhao/alexnet.dense.cpu.pb'
# pb_path='/home/chenhao/alexnet.dense.mlu.pb'
# pbtxt_path='cambrain_mlu.txt'

def main(args):
    graphdef_to_pbtxt(args.pb_path, args.pbtxt_path)


def graphdef_to_pbtxt(filename, pbtxt_path):
    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', pbtxt_path, as_text=True)
    return


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('pb_path', type=str,
                        help='protobuf file to read.')
    parser.add_argument('pbtxt_path', type=str,
                        help='dest file path.')
    parser.add_argument('--image_files', type=str, nargs='+', help='Images')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import argparse


def main(args):
    a = args.image_files
    b = args.first_conv
    print a, b


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_files', type=str, nargs='+', help='Images')
    parser.add_argument('first_conv', type=int,
                        help='Use first conv2d or not.1:use 0:not use')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

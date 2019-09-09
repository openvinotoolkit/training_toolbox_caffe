#!/usr/bin/env python3

"""
 Copyright (c) 2018 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import caffe

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Age-gender model')
    parser.add_argument('--gpu', help='GPU id to use', default=0, type=int)
    parser.add_argument('--compute_mode', type=str, choices=['CPU', 'GPU'], default='GPU',
                        help='Caffe compute mode: CPU or GPU')
    parser.add_argument('--def', dest='prototxt', help='prototxt file defining the network',
                        required=True, type=str)
    parser.add_argument('--net', dest='caffemodel', help='model to test', required=True, type=str)
    parser.add_argument('--gt', dest='gt', help='Path to groundtruth annotation txt file', type=str, required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()

def preprocess_image(img, size):
    """Transforms input image into network-compatible format.
    :param img: Input image
    :param size: Target size of network input
    :return: Network-compatible input blob
    """

    img = cv2.resize(img, size)
    return img.transpose((2, 0, 1)).astype(np.float32)

def main():
    """
    """
    args = parse_args()

    if args.compute_mode == 'GPU':
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
    if args.compute_mode == 'CPU':
        caffe.set_mode_cpu()

    print('Called with args:')
    print(args)

    assert os.path.exists(args.gt)
    assert os.path.exists(args.prototxt)
    assert os.path.exists(args.caffemodel)

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    data_dir = os.path.dirname(args.gt)
    with open(args.gt) as f:
        lines = f.readlines()

    gender_positives = 0
    gender_negatives = 0
    age_dif = 0
    num = 0

    for line in lines:
        path, l_gen, l_age = line.split()
        gen = bool(float(l_gen))
        age = float(l_age)
        print('path = ', path, 'gen = ', gen, 'age = ', age)

        # image reading
        im_path = os.path.join(data_dir, path)
        if not os.path.exists(im_path):
            print("Can't load", im_path)
            continue
        img = cv2.imread(im_path)

        if img is None:
            print("Can't load", im_path)
            continue

        # image resizing
        in_height, in_width = net.blobs['data'].data.shape[2:]
        net.blobs['data'].data[...] = np.array(preprocess_image(img, (in_width, in_height)).astype(np.float32)/255.)

        gender_out_layer = 'prob'
        age_out_layer = 'fc3_a' #'age_conv3'

        # forward pass
        net.forward()
        female_prob = net.blobs[gender_out_layer].data[0][0][0][0]
        male_prob = net.blobs[gender_out_layer].data[0][1][0][0]
        age_output = net.blobs[age_out_layer].data[0][0][0][0] * 100

        print("maleProb:", male_prob, "age:", age_output)

        if bool(male_prob > 0.5) == gen:
            gender_positives += 1
        else:
            gender_negatives += 1
        age_dif += abs(age_output - age*100)
        num += 1

    gender_accuracy = gender_positives / (gender_positives + gender_negatives)
    print('gender_accuracy = ', gender_accuracy)
    age_mae = age_dif/num
    print('age mae = ', age_mae)


if __name__ == '__main__':
    main()

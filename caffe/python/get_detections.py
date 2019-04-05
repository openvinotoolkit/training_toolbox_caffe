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

import colorsys
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import caffe

from utils.dataset_xml_reader import read_annotation, write_annotation, ImageAnnotation


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Face detection model')
    parser.add_argument('--gpu', help='GPU id to use', default=0, type=int)
    parser.add_argument('--compute_mode', type=str, choices=['CPU', 'GPU'], default='GPU',
                        help='Caffe compute mode: CPU or GPU')
    parser.add_argument('--def', dest='prototxt', help='prototxt file defining the network',
                        required=True, type=str)
    parser.add_argument('--net', dest='caffemodel', help='model to test', required=True, type=str)
    parser.add_argument('--gt', help='Path to groundtruth annotation (xml)', type=str, required=True)
    parser.add_argument('--det', help='Path to output detections (xml)', type=str, required=True)
    parser.add_argument('--det_out_name', type=str, default='detection_out', help='Name of detection output layer.')
    parser.add_argument('--resize_to', type=str, default='-1x-1',
                        help='Resize image to WxH, if W or H is -1 then image is not resized.')
    parser.add_argument('--means', type=str, default='0,0,0', help='RGB mean values to be subtracted from input data')
    parser.add_argument('--scale', type=float, default='1', help='RGB mean values to be subtracted from input data')
    parser.add_argument('--thr', type=float, default=0.0, help='Confidence threshold')
    parser.add_argument('--delay', type=int, default=-1,
                        help='cv2.waitKey value, if it is -1 then visualization is disabled.')
    parser.add_argument('--labels', type=str, required=True,
                        help="list of labels, e.g. : ['background', 'pedestrian', 'face'])")
    parser.add_argument('--target_label', type=str, default=None, help='Other labels are filtered out')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def preprocess_image(img, resize_to, means, scale):
    """
    Preprocess image
    :param resize_to: output size
    :param mean: mean
    :param scale: scale
    :return: preprocessed image
    """
    if resize_to[0] > 0 and resize_to[1] > 0:
        im_resized = cv2.resize(img, resize_to)
    else:
        im_resized = img

    im_resized = im_resized.astype(np.float32)

    im_resized[:, :, 0] -= means[0]
    im_resized[:, :, 1] -= means[1]
    im_resized[:, :, 2] -= means[2]

    im_resized *= scale

    im_transposed = im_resized.transpose((2, 0, 1))
    return im_transposed


def main():
    """
    Get found detections and store them in output xml file
    """
    args = parse_args()

    if args.compute_mode == 'GPU':
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
    if args.compute_mode == 'CPU':
        caffe.set_mode_cpu()

    print('Called with args:')
    print(args)

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    resize_to = tuple([int(x) for x in args.resize_to.split('x')])
    means = tuple([float(x) for x in args.means.split(',')])

    scale = args.scale

    print('reading annotation...')
    annotation = read_annotation(args.gt)

    detections = dict()

    detection_output_layer_name = args.det_out_name

    labels = eval(args.labels)

    visualize = args.delay > -1

    if visualize:
        num_labels = len(labels)
        hsv_tuples = [(x * 1.0 / num_labels, 0.5, 0.5) for x in range(num_labels)]
        rgb_tuples = [colorsys.hsv_to_rgb(*x) * 255 for x in hsv_tuples]
        colors = [(255 * x[0], 255 * x[1], 255 * x[2]) for x in rgb_tuples]

    print('running detector')

    times = []
    for frame in tqdm(annotation):

        # image reading
        image_path = annotation[frame].image_path
        img = cv2.imread(image_path)

        if img is None:
            print("Can't load", image_path, file=sys.stderr)

        # image resizing
        orig_size = (img.shape[1], img.shape[0])
        im_transposed = preprocess_image(img, resize_to, means, scale)
        net.blobs['data'].reshape(1, 3, im_transposed.shape[1], im_transposed.shape[2])

        # forward pass
        start = time.time()
        res = net.forward(data=np.array([im_transposed]))[detection_output_layer_name][0][0]
        end = time.time()
        times += [end - start]

        frame_detections = ImageAnnotation(image_path, [])

        for det in res:
            _, label, score, xmin, ymin, xmax, ymax = det

            label = int(label)

            if score < args.thr:
                continue

            if args.target_label is not None and labels[label] != args.target_label:
                continue

            xmin *= orig_size[0]
            xmax *= orig_size[0]
            ymin *= orig_size[1]
            ymax *= orig_size[1]

            rect = ' '.join([str(int(round(x))) for x in [xmin, ymin, xmax - xmin, ymax - ymin]])
            obj = {'bbox': rect, 'score': score, 'type': labels[label]}

            frame_detections.objects += [obj]

            if visualize:
                if score > 0.0:
                    color = colors[label]
                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    text = labels[label][:3] + ": " + str(round(score*100) * 0.01)
                    cv2.putText(img, text, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

        detections[frame] = frame_detections

        if visualize:
            cv2.imshow('image', img)
            cv2.waitKey(args.delay)

    print('Mean forward pass time', np.mean(times))

    write_annotation(detections, args.det)


if __name__ == '__main__':
    main()

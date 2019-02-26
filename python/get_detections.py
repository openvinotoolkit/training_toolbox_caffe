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

from utils.dataset_xml_reader import read_annotation, write_annotation, ImageAnnotation

import argparse
import os
import cv2
import sys
import numpy as np
import caffe

from tqdm import tqdm
import colorsys
import time


def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Face detection model')
    parser.add_argument('--gpu', help='GPU id to use', default=0, type=int)
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

    args = parser.parse_args()
    return args

def preprocess_image(im, resize_to, means, scale):

    if resize_to[0] > 0 and resize_to[1] > 0:
        im_resized = cv2.resize(im, resize_to)
    else:
        im_resized = im

    im_resized = im_resized.astype(np.float32)

    im_resized[:, :, 0] -= means[0]
    im_resized[:, :, 1] -= means[1]
    im_resized[:, :, 2] -= means[2]

    im_resized *= scale

    im_transposed = im_resized.transpose((2, 0, 1))
    return im_transposed


if __name__ == '__main__':
    args = parse_args()
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

    print('Called with args:')
    print(args)

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    resize_to = tuple([int(x) for x in args.resize_to.split('x')])
    means = tuple([float(x) for x in args.means.split(',')])

    scale = args.scale

    print ('reading annotation...')
    annotation = read_annotation(args.gt)

    detections = dict()

    detection_output_layer_name = args.det_out_name

    labels = eval(args.labels)

    visualize = args.delay > -1

    if visualize:
        N = len(labels)
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x) * 255, HSV_tuples)
        colors = [(255 * x[0], 255* x[1], 255 * x[2]) for x in RGB_tuples]

    print ('running detector')

    times = []
    for frame in tqdm(annotation):

        # image reading
        image_path = annotation[frame].image_path
        im = cv2.imread(image_path)

        if im is None:
            print >> sys.stderr, "Can't load", image_path

        # image resizing
        orig_size = (im.shape[1], im.shape[0])
        im_transposed = preprocess_image(im, resize_to, means, scale)
        net.blobs['data'].reshape(1, 3, im_transposed.shape[1], im_transposed.shape[2])

        # forward pass
        start = time.time()
        res = net.forward(data=np.array([im_transposed]))[detection_output_layer_name][0][0]
        end = time.time()
        times += [end - start]


        frame_detections = ImageAnnotation(image_path, [])

        for det in res:
            _, label, score, xmin, ymin, xmax, ymax  = det

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
            object = {'bbox': rect, 'score': score, 'type': labels[label]}

            frame_detections.objects += [object]

            if visualize:
                if score > 0.0:
                    color = colors[label]
                    cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    text = labels[label][:3] + ": " + str(round(score*100) * 0.01)
                    cv2.putText(im, text, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

        detections[frame]=frame_detections

        if visualize:
            cv2.imshow('image', im)
            cv2.waitKey(args.delay)

    print 'Mean forward pass time', np.mean(times)

    write_annotation(detections, args.det)

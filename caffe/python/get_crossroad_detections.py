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


import argparse
import time
from collections import defaultdict
from contextlib import contextmanager
from six import iteritems
from tqdm import tqdm

import json
import numpy as np
import cv2
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2

from utils.image_grabber import IterableImageGrabberInterface

#pylint: disable=no-member
def load_labels(label_map_file_path):
    """ Load labels from labelmap file

    :param label_map_file_path: Labelmap file
    """
    label_map = caffe_pb2.LabelMap()
    label_map_serialized = open(label_map_file_path, 'rt').read()
    text_format.Merge(str(label_map_serialized), label_map)
    labels = [str(item.display_name) for item in label_map.item]
    return labels


def get_detection(raw_detection, original_frame_sizes):
    """ Get detection fields from the net's output
    """
    assert raw_detection.size == 7, 'Wrong number of elements in raw detector output'
    image_id = int(raw_detection[0])
    class_id = int(raw_detection[1])
    score = float(raw_detection[2])
    bbox = np.clip(raw_detection[3:7], 0, 1)
    bbox *= np.tile(original_frame_sizes[image_id], 2)
    return image_id, class_id, score, bbox


class Timer(object):
    """ Timer class
    """
    def __init__(self):
        super(Timer, self).__init__()
        self.timers = defaultdict(lambda: [0.0, 0.0])

    @contextmanager
    def timeit_context(self, name):
        """ Set timers
        """
        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        self.timers[name][0] += elapsed_time
        self.timers[name][1] += 1

    def print_stat(self):
        """ Print timing
        """
        print('timing:')
        for name, (total_time, number_of_calls) in iteritems(self.timers):
            if number_of_calls > 0:
                print('\t{}: {:>10.2f} ms per call, {:>10.2f} s total'.format(name,
                                                                              total_time / number_of_calls * 1000,
                                                                              total_time))


def preprocess_image(frame_src, input_width, input_height, mean, scale):
    """ Preprocess image
    """
    frame_size = frame_src.shape[:2]
    frame = cv2.resize(frame_src, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    frame = frame.astype(np.float32, copy=False)
    frame -= mean
    frame *= scale
    frame = frame.transpose((2, 0, 1))[np.newaxis, ...]
    return frame, frame_size[::-1]

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Script for running SSD detector on images/video',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('prototxt', help='Path to prototxt model description')
    parser.add_argument('caffemodel', help='Path to caffemodel binary weights file')
    parser.add_argument('labelmap', help='Path to file with label names')
    parser.add_argument('source_type', choices=('video', 'imdb', 'annotation'), default='video',
                        help='Image source type')
    parser.add_argument('source', help='Path to images source')

    parser.add_argument('--annotation_out', dest='annotation_out_file_path', default=None,
                        help='Path to annotation file to save detections to')
    parser.add_argument('-t', '--threshold', dest='confidence_threshold', default=0.05, type=float,
                        help='confidence threshold to filter out weak detections')
    parser.add_argument('--gpu', help='GPU id to use', default=0, type=int)
    parser.add_argument('--compute_mode', type=str, choices=['CPU', 'GPU'], default='GPU',
                        help='Caffe compute mode: CPU or GPU')
    parser.add_argument('--mean', nargs=3, default=(104, 117, 123), type=float,
                        help='Pixel mean value (BGR format) to subtracted before feeding the image to the net')
    parser.add_argument('--scale', default=1.0, type=float,
                        help='')
    parser.add_argument('--batch', default=1, type=int,
                        help='')
    parser.add_argument('--output_blob', dest='output_blob_name', default='detection_out',
                        help='Name of network output blob')
    return parser.parse_args()

def main():
    """ Get detections from the net's output and store them in xml file for further evaluation.
    """
    args = parse_args()

    if args.compute_mode == 'GPU':
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
    if args.compute_mode == 'CPU':
        caffe.set_mode_cpu()

    assert args.batch > 0, 'Batch size should be >= 1.'

    net = caffe.Net(args.prototxt, caffe.TEST, weights=args.caffemodel)
    assert len(net.inputs) == 1, 'Single input blob is expected'
    input_data_shape = net.blobs[net.inputs[0]].data.shape
    assert input_data_shape[0] == args.batch, 'Requested batch size does not match the one defined in prototxt'
    assert input_data_shape[1] == 3, 'Color image is expected'
    input_width = input_data_shape[3]
    input_height = input_data_shape[2]

    class_labels = load_labels(args.labelmap)
    batch = []
    annotation = []
    timer = Timer()
    do_exit = False

    print("args.source = ", args.source, "args.source_type, ", args.source_type)
    with IterableImageGrabberInterface.create_iterable_image_grabber(args.source_type, args.source) as data_provider:
        for frame_src, image_name in tqdm(data_provider):
            with timer.timeit_context('preprocessing'):
                frame, frame_size = preprocess_image(frame_src, input_width, input_height, args.mean, args.scale)
                batch.append({'raw_image': frame_src,
                              'image': frame,
                              'size': frame_size,
                              'name': image_name,
                              'objects': []})

            if len(batch) == args.batch:
                with timer.timeit_context('forward pass'):
                    output_blobs = net.forward(data=np.concatenate([item['image'] for item in batch], axis=0))

                with timer.timeit_context('postprocessing'):
                    assert args.output_blob_name in output_blobs, \
                        'The net has multiple output blobs and none of them has name "detection_out"'
                    detections = output_blobs[args.output_blob_name].reshape(-1, 7)

                    for detection in detections:
                        image_id, class_id, score, bbox = get_detection(detection, [item['size'] for item in batch])
                        if score > args.confidence_threshold:
                            batch[image_id]['objects'].append({'label': class_labels[class_id],
                                                               'bbox': bbox.tolist(),
                                                               'score': score})

                    for batch_item in batch:
                        annotation.append({'image': batch_item['name'],
                                           'objects': batch_item['objects']
                                          })

                    batch = []
                    if do_exit:
                        break

    timer.print_stat()

    if args.annotation_out_file_path is not None:
        json.dump(annotation, open(args.annotation_out_file_path, 'w'))

if __name__ == "__main__":
    main()

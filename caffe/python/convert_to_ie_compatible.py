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

#pylint: disable=ungrouped-imports
import os

from argparse import ArgumentParser
from builtins import range
from os.path import exists

import numpy as np

os.environ['GLOG_minloglevel'] = '2'
#pylint: disable=wrong-import-position
import caffe


def load_centers(network, name, eps):
    """Load values of centers from the specified network by name.

    :param network: Network to load center values
    :param name: Name of parameter with centers
    :return: Normalized centers
    """

    assert name in network.params.keys(), 'Cannot find name: {} in params'.format(name)

    params = network.params[name]
    assert len(params) == 1

    centers = params[0].data

    norms = np.sqrt(np.sum(np.square(centers), axis=1, keepdims=True) + eps)
    normalized_centers = centers / norms

    return normalized_centers


def convert_to_conv_params(centers_data):
    """Converts input data to convolution compatible format of parameters.

    :param centers_data: Input data
    :return: Parameters for convolution layer
    """

    assert len(centers_data.shape) == 2

    num_centers = centers_data.shape[0]
    embedding_size = centers_data.shape[1]

    return centers_data.reshape([num_centers, embedding_size, 1, 1])


def main():
    """Converts input representation of network to IE-compatible format.
    """

    parser = ArgumentParser()
    parser.add_argument('--in_proto', '-m', type=str, required=True, help='Input .prototxt file')
    parser.add_argument('--in_weights', '-w', type=str, required=True, help='Input .caffemodel file')
    parser.add_argument('--out_weights', '-o', type=str, required=True, help='Output .caffemodel file')
    parser.add_argument('--centers_name', '-c', type=str, default='centers/action/params',
                        help='Name of blob with centers')
    parser.add_argument('--num_anchors', '-n', type=int, default=4, help='Output .caffemodel file')
    parser.add_argument('--out_prefix', '-t', type=str, default='logits/anchor',
                        help='Template name of output blobs')
    parser.add_argument('--eps', '-e', type=float, required=False, default=1e-8, help='Normalization epsilon')
    args = parser.parse_args()

    assert exists(args.in_proto)
    assert exists(args.in_weights)

    net = caffe.Net(args.in_proto, args.in_weights, caffe.TEST)

    centers = load_centers(net, args.centers_name, args.eps)
    conv_params = convert_to_conv_params(centers)

    for i in range(args.num_anchors):
        layer_name = '{}{}'.format(args.out_prefix, i + 1)
        layer_params = net.params[layer_name]
        assert len(layer_params) == 1

        layer_params[0].data[...] = conv_params

    net.save(args.out_weights)


if __name__ == '__main__':
    main()

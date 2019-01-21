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

from argparse import ArgumentParser

from caffe import set_mode_cpu
from caffe.proto import caffe_pb2


def main():
    """Adds specified prefix to name of network layers.
    """

    parser = ArgumentParser()
    parser.add_argument('--weights_in', '-i', type=str, required=True)
    parser.add_argument('--weights_out', '-o', type=str, required=True)
    parser.add_argument('--prefix', '-p', type=str, default='cl/')
    args = parser.parse_args()

    set_mode_cpu()

    net_weights = caffe_pb2.NetParameter()
    with open(args.weights_in, 'rb') as binary_proto_file:
        binary_message = binary_proto_file.read()
        net_weights.ParseFromString(binary_message)

    #pylint: disable=no-member
    layers = net_weights.layer
    if not layers:
        layers = net_weights.layers

    layer_names = [layer.name for layer in layers]
    for layer, name in zip(layers, layer_names):
        for i, top in enumerate(layer.top):
            layer.top[i] = args.prefix + top
        for i, bottom in enumerate(layer.bottom):
            layer.bottom[i] = args.prefix + bottom
        layer.name = args.prefix + name

    with open(args.weights_out, 'wb') as file_stream:
        file_stream.write(net_weights.SerializeToString())

if __name__ == '__main__':
    main()

# pylint: skip-file

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
import tempfile
import unittest

import caffe
import numpy as np


def python_net_file():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("""name: 'adaptive_weighting_loss_layer_test_net'
                   force_backward: true
                   layer { name: "input1" type: "Input" top: "input1"
                           input_param { shape { dim: 1 } } }
                   layer { name: "input2" type: "Input" top: "input2"
                           input_param { shape { dim: 1 } } }
                   layer { name: "input3" type: "Input" top: "input3"
                           input_param { shape { dim: 1 } } }
                   layer { type: 'Python' name: 'adaptive_loss'
                           bottom: 'input1' bottom: 'input2' bottom: 'input3'
                           top: 'adaptive_loss_value'
                           python_param { module: 'caffe.custom_layers.adaptive_weighting_loss_layer'
                                          layer: 'AdaptiveWeightingLossLayer'
                                          param_str: '{ "scale": 1.0, "init": 0.0}' } }""")
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(), 'Caffe built without Python layer support')
class TestScheduleScaleLayer(unittest.TestCase):
    def setUp(self):
        net_file = python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        data = np.random.uniform(0.5, 15.0, size=3)

        target_value = np.sum(data)

        self.net.blobs['input1'].data[...] = data[0]
        self.net.blobs['input2'].data[...] = data[1]
        self.net.blobs['input3'].data[...] = data[2]

        net_outputs = self.net.forward()
        predicted_value = net_outputs['adaptive_loss_value']

        self.assertEqual(len(predicted_value), 1)
        self.assertAlmostEqual(predicted_value[0], target_value, places=5)

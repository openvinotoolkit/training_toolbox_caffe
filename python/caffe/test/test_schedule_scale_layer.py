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
        f.write("""name: 'schedule_scale_layer_test_net'
                   force_backward: true
                   layer { name: "input" type: "Input" top: "input"
                           input_param { shape { dim: 2 dim: 7 dim: 3 dim: 5 } } }
                   layer { type: 'Python' name: 'scheduled_scale' bottom: 'input' top: 'scheduled_scale_values'
                           python_param { module: 'caffe.custom_layers.schedule_scale_layer'
                                          layer: 'ScheduledScaleLayer'
                                          param_str: '{ "target_scale": 1.5, "use_last": True}' } }""")
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(), 'Caffe built without Python layer support')
class TestScheduleScaleLayer(unittest.TestCase):
    def setUp(self):
        net_file = python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

        self.data = np.random.normal(size=[2, 7, 3, 5])

    def test_forward(self):
        target_values = 1.5 * self.data

        self.net.blobs['input'].data[...] = self.data

        net_outputs = self.net.forward()
        predicted_values = net_outputs['scheduled_scale_values']
        self.assertEqual(predicted_values.shape, target_values.shape)

        target_values = target_values.flatten()
        predicted_values = predicted_values.flatten()

        for i in xrange(len(target_values)):
            self.assertAlmostEqual(predicted_values[i], target_values[i], places=5)

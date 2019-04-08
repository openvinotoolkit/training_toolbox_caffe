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

from builtins import range

import caffe
import numpy as np


def python_net_file():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("""name: 'actions_detection_output_layer_test_net'
                   force_backward: true
                   layer { name: "in_detections" type: "Input" top: "in_detections"
                           input_param { shape { dim: 1 dim: 1 dim: 2 dim: 10 } } }
                   layer { name: "anchor1" type: "Input" top: "anchor1"
                           input_param { shape { dim: 2 dim: 3 dim: 4 dim: 2 } } }
                   layer { name: "anchor2" type: "Input" top: "anchor2"
                           input_param { shape { dim: 2 dim: 3 dim: 4 dim: 2 } } }
                   layer { type: 'Python' name: 'actions_detection_output' bottom: 'in_detections'
                           bottom: 'anchor1' bottom: 'anchor2' top: 'out_detections'
                           python_param { module: 'caffe.custom_layers.actions_detection_output_layer'
                                          layer: 'ActionsDetectionOutputLayer'
                                          param_str: '{ "num_anchors": 2 }' } }""")
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(), 'Caffe built without Python layer support')
class TestActionsDetectionOutputLayer(unittest.TestCase):
    def setUp(self):
        net_file = python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

        self.in_detections = np.array([[0, 1, 1, 0, 0, 1, 1, 0, 2, 1],
                                       [1, 1, 1, 0, 0, 1, 1, 1, 3, 2]], dtype=np.float32)

        anchor1 = np.random.uniform(size=[2, 3, 4, 2])
        anchor1 /= np.sum(anchor1, axis=1, keepdims=True)
        anchor2 = np.random.uniform(size=[2, 3, 4, 2])
        anchor2 /= np.sum(anchor2, axis=1, keepdims=True)

        trg_class_conf = [0.0, 1.0]
        anchor1[0, 1, 2, :] = trg_class_conf
        anchor2[1, 2, 3, :] = trg_class_conf

        self.anchors = [anchor1, anchor2]

    def test_forward(self):
        self.net.blobs['in_detections'].data[...] = self.in_detections
        self.net.blobs['anchor1'].data[...] = self.anchors[0]
        self.net.blobs['anchor2'].data[...] = self.anchors[1]

        net_outputs = self.net.forward()
        out_detections = net_outputs['out_detections']
        self.assertTupleEqual(out_detections.shape, (1, 1, 2, 8))

        for i in range(2):
            in_detection = self.in_detections[i]
            out_detection = out_detections[0, 0, i]

            self.assertEqual(int(in_detection[0]), int(out_detection[0]))
            self.assertAlmostEqual(float(in_detection[2]), float(out_detection[2]), places=5)
            self.assertEqual(int(out_detection[1]), 1)
            self.assertEqual(int(in_detection[3]), int(out_detection[4]))
            self.assertEqual(int(in_detection[4]), int(out_detection[5]))
            self.assertEqual(int(in_detection[5]), int(out_detection[6]))
            self.assertEqual(int(in_detection[6]), int(out_detection[7]))

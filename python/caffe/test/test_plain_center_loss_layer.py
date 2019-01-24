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
        f.write("""name: 'plain_center_loss_layer_test_net'
                   force_backward: true
                   layer { name: "centers" type: "Input" top: "centers"
                           input_param { shape { dim: 2 dim: 7 } } }
                   layer { name: "embeddings" type: "Input" top: "embeddings"
                           input_param { shape { dim: 3 dim: 7 } } }
                   layer { name: "labels" type: "Input" top: "labels"
                           input_param { shape { dim: 3 } } }
                   layer { type: 'Python' name: 'plain_center_loss' bottom: 'centers'
                           bottom: 'embeddings' bottom: 'labels' top: 'plain_center_loss_value'
                           python_param { module: 'caffe.custom_layers.plain_center_loss_layer'
                                          layer: 'PlainCenterLossLayer'
                                          param_str: '{ "num_anchors": 2, "valid_action_ids": [0, 1]}' } }""")
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(), 'Caffe built without Python layer support')
class TestPlainCenterLossLayer(unittest.TestCase):
    def setUp(self):
        net_file = python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

        self.centers = np.random.normal(size=[2, 7])
        self.centers /= np.sqrt(np.sum(np.square(self.centers), axis=1, keepdims=True))

        self.embeddings = np.random.normal(size=[3, 7])
        self.embeddings /= np.sqrt(np.sum(np.square(self.embeddings), axis=1, keepdims=True))

        self.labels = np.array([0, 1, 0], dtype=np.float32)

    def test_forward(self):
        trg_loss_value = 0.0
        for i in range(3):
            class_id = int(self.labels[i])

            dist = 1.0 - np.sum(self.centers[class_id] * self.embeddings[i])
            trg_loss_value += dist / 3.0

        self.net.blobs['centers'].data[...] = self.centers
        self.net.blobs['embeddings'].data[...] = self.embeddings
        self.net.blobs['labels'].data[...] = self.labels

        net_outputs = self.net.forward()
        predicted_loss_value = net_outputs['plain_center_loss_value']
        self.assertEqual(len(predicted_loss_value), 1)

        self.assertAlmostEqual(predicted_loss_value[0], trg_loss_value, places=5)

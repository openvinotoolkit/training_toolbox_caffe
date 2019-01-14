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
        f.write("""name: 'local_push_loss_layer_test_net'
                   force_backward: true
                   layer { name: "detections" type: "Input" top: "detections"
                           input_param { shape { dim: 1 dim: 1 dim: 3 dim: 11 } } }
                   layer { name: "centers" type: "Input" top: "centers"
                           input_param { shape { dim: 2 dim: 5 } } }
                   layer { name: "anchor1" type: "Input" top: "anchor1"
                           input_param { shape { dim: 2 dim: 5 dim: 3 dim: 4 } } }
                   layer { name: "anchor2" type: "Input" top: "anchor2"
                           input_param { shape { dim: 2 dim: 5 dim: 3 dim: 4 } } }
                   layer { type: 'Python' name: 'local_push_loss' bottom: 'detections' bottom: 'centers'
                           bottom: 'anchor1' bottom: 'anchor2' top: 'local_push_loss_value'
                           python_param { module: 'caffe.custom_layers.local_push_loss_layer'
                                          layer: 'LocalPushLossLayer'
                                          param_str: '{ "num_anchors": 2, "valid_action_ids": [0, 1], "margin": 0.6}' } }""")
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(), 'Caffe built without Python layer support')
class TestLocalPushLossLayer(unittest.TestCase):
    def setUp(self):
        net_file = python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

        self.detections = np.array([[0, 1, 0, 0, 1, 1, 0, 0, 0, 2, 1],
                                    [1, 1, 0, 0, 1, 1, 1, 1, 1, 3, 2],
                                    [1, 1, 0, 0, 1, 1, 1, 2, 0, 0, 2]], dtype=np.float32)

        self.centers = np.random.normal(size=2 * 5).reshape([2, 5])
        self.centers /= np.sqrt(np.sum(np.square(self.centers), axis=1, keepdims=True))

        anchor1 = np.random.normal(size=[2, 5, 3, 4])
        anchor1 /= np.sqrt(np.sum(np.square(anchor1), axis=1, keepdims=True))
        anchor2 = np.random.normal(size=[2, 5, 3, 4])
        anchor2 /= np.sqrt(np.sum(np.square(anchor2), axis=1, keepdims=True))
        self.anchors = [anchor1, anchor2]

    def test_forward(self):
        local_loss_values = []
        for i in xrange(3):
            detection = self.detections[i]
            item = int(detection[0])
            anchor_id = int(detection[6])
            class_id = int(detection[8])
            x_pos = int(detection[9])
            y_pos = int(detection[10])

            det_embedding = self.anchors[anchor_id][item, :, y_pos, x_pos]
            center_embedding = self.centers[class_id]

            pos_distance = 1.0 - np.sum(det_embedding * center_embedding)

            for j in xrange(2):
                if j == class_id:
                    continue

                neg_distance = 1.0 - np.sum(det_embedding * self.centers[j])

                local_loss = 0.6 + pos_distance - neg_distance
                if local_loss > 0.0:
                    local_loss_values.append(local_loss)

        trg_loss_value = np.mean(local_loss_values) if len(local_loss_values) > 0 else 0.0

        self.net.blobs['detections'].data[...] = self.detections
        self.net.blobs['centers'].data[...] = self.centers
        self.net.blobs['anchor1'].data[...] = self.anchors[0]
        self.net.blobs['anchor2'].data[...] = self.anchors[1]

        net_outputs = self.net.forward()
        predicted_loss_value = net_outputs['local_push_loss_value']
        self.assertEqual(len(predicted_loss_value), 1)

        self.assertAlmostEqual(predicted_loss_value[0], trg_loss_value, places=5)

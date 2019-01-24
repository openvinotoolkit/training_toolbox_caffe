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
        f.write("""name: 'split_loss_layer_test_net'
                   force_backward: true
                   layer { name: "detections" type: "Input" top: "detections"
                           input_param { shape { dim: 1 dim: 1 dim: 2 dim: 11 } } }
                   layer { name: "centers" type: "Input" top: "centers"
                           input_param { shape { dim: 2 dim: 5 } } }
                   layer { name: "anchor1" type: "Input" top: "anchor1"
                           input_param { shape { dim: 2 dim: 5 dim: 3 dim: 4 } } }
                   layer { name: "anchor2" type: "Input" top: "anchor2"
                           input_param { shape { dim: 2 dim: 5 dim: 3 dim: 4 } } }
                   layer { type: 'Python' name: 'split_loss' bottom: 'detections' bottom: 'centers'
                           bottom: 'anchor1' bottom: 'anchor2' top: 'split_loss_value'
                           python_param { module: 'caffe.custom_layers.split_loss_layer'
                                          layer: 'SplitLossLayer'
                                          param_str: '{ "num_anchors": 2, "valid_action_ids": [0, 1], "margin": 0.6}' } }""")
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(), 'Caffe built without Python layer support')
class TestSplitLossLayer(unittest.TestCase):
    def setUp(self):
        net_file = python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

        self.detections = np.array([[0, 1, 0, 0, 1, 1, 0, 0, 0, 2, 1],
                                    [0, 1, 0, 0, 1, 1, 1, 1, 1, 3, 2]], dtype=np.float32)

        self.centers = np.random.normal(size=[2, 5])
        self.centers /= np.sqrt(np.sum(np.square(self.centers), axis=1, keepdims=True))

        anchor1 = np.random.normal(size=[2, 5, 3, 4])
        anchor1 /= np.sqrt(np.sum(np.square(anchor1), axis=1, keepdims=True))
        anchor2 = np.random.normal(size=[2, 5, 3, 4])
        anchor2 /= np.sqrt(np.sum(np.square(anchor2), axis=1, keepdims=True))

        # create incorrect labels
        anchor1[0, :, 1, 2] = self.centers[0]
        anchor2[0, :, 2, 3] = self.centers[0]

        self.anchors = [anchor1, anchor2]

    def test_forward(self):
        embeddings = []
        for i in range(2):
            detection = self.detections[i]
            item = int(detection[0])
            anchor_id = int(detection[6])
            x_pos = int(detection[9])
            y_pos = int(detection[10])

            embeddings.append(self.anchors[anchor_id][item, :, y_pos, x_pos])

        anchor_embed = embeddings[0]
        ref_embed = embeddings[1]

        embed_dist = 1.0 - np.sum(anchor_embed * ref_embed)
        ref_dist_to_center = 1.0 - np.sum(self.centers[1] * ref_embed)

        loss = 0.6 + ref_dist_to_center - embed_dist
        trg_loss_value = loss if loss > 0.0 else 0.0

        self.net.blobs['detections'].data[...] = self.detections
        self.net.blobs['centers'].data[...] = self.centers
        self.net.blobs['anchor1'].data[...] = self.anchors[0]
        self.net.blobs['anchor2'].data[...] = self.anchors[1]

        net_outputs = self.net.forward()
        predicted_loss_value = net_outputs['split_loss_value']
        self.assertEqual(len(predicted_loss_value), 1)

        self.assertAlmostEqual(predicted_loss_value[0], trg_loss_value, places=5)

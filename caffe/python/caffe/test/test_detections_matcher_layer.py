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
        f.write("""name: 'detections_matcher_layer_test_net'
                   force_backward: true
                   layer { name: "pred_detections" type: "Input" top: "pred_detections"
                           input_param { shape { dim: 1 dim: 1 dim: 3 dim: 10 } } }
                   layer { name: "gt_detections" type: "Input" top: "gt_detections"
                           input_param { shape { dim: 1 dim: 1 dim: 2 dim: 8 } } }
                   layer { name: "prior_boxes" type: "Input" top: "prior_boxes"
                           input_param { shape { dim: 1 dim: 2 dim: 96} } }
                   layer { type: 'Python' name: 'detections_matcher' bottom: 'pred_detections'
                           bottom: 'gt_detections' bottom: 'prior_boxes' top: 'matched_detections'
                           python_param { module: 'caffe.custom_layers.detections_matcher_layer'
                                          layer: 'DetMatcherLayer'
                                          param_str: '{ "num_anchors": 2, "height": 3, "width": 4, "valid_action_ids": [0, 1, 2] }' } }""")
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(), 'Caffe built without Python layer support')
class TestDetectionsMatcherLayer(unittest.TestCase):
    def setUp(self):
        net_file = python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

        self.pred_detections = np.array([[0., 1., 1., 0.22, 0.31, 0.25, 0.35, 0., 2., 1.],
                                         [1., 1., 1., 0.51, 0.45, 0.61, 0.72, 1., 3., 2.],
                                         [1., 1., 1., 0.73, 0.22, 0.98, 0.34, 1., 0., 2.]], dtype=np.float32)

        self.gt_detections = np.array([[0., 1., 0., 0.2, 0.3, 0.25, 0.35, 0.],
                                       [1., 0., 1., 0.7, 0.2, 0.99, 0.35, 0.]], dtype=np.float32)

        self.prior_boxes = np.random.uniform(size=[1, 2, 3, 4, 2, 4])
        self.prior_boxes[0, 0, 1, 2, 0, :] = [0.2, 0.3, 0.25, 0.35]
        self.prior_boxes[0, 0, 2, 0, 1, :] = [0.7, 0.2, 0.99, 0.35]
        self.prior_boxes = self.prior_boxes.reshape([1, 2, 3 * 4 * 2 * 4])

    def test_forward(self):
        self.net.blobs['pred_detections'].data[...] = self.pred_detections
        self.net.blobs['gt_detections'].data[...] = self.gt_detections
        self.net.blobs['prior_boxes'].data[...] = self.prior_boxes

        net_outputs = self.net.forward()
        matched_detections = net_outputs['matched_detections']
        self.assertTupleEqual(matched_detections.shape, (1, 1, 2, 11))

        gt_detection_record = [[0., 1., 0.22, 0.31, 0.25, 0.35, 0., 0., 1., 2., 1.],
                               [1., 1., 0.73, 0.22, 0.98, 0.34, 1., 1., 0., 0., 2.]]
        for i in range(2):
            for j in range(11):
                self.assertAlmostEqual(matched_detections[0, 0, i, j], gt_detection_record[i][j], places=6)

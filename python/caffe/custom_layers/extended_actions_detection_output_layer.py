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

import numpy as np
import traceback
from collections import namedtuple

from caffe._caffe import log as LOG
from caffe._caffe import Layer as BaseLayer


InputDetection = namedtuple('InputDetection', 'item_id, det_conf, anchor_id,'
                                              'xmin, ymin, xmax, ymax,'
                                              'x_pos, y_pos')
OutputDetection = namedtuple('OutputDetection', 'item_id, action,'
                                                'det_conf, action_conf,'
                                                'xmin, ymin, xmax, ymax,'
                                                'anchor_id, x_pos, y_pos')

INPUT_RECORD_SIZE = 10
OUTPUT_RECORD_SIZE = 11


class ExtendedActionsDetectionOutputLayer(BaseLayer):
    @staticmethod
    def _translate_prediction(record):
        bbox = InputDetection(item_id=int(record[0]),
                              det_conf=float(record[2]),
                              anchor_id=int(record[7]),
                              xmin=float(record[3]),
                              ymin=float(record[4]),
                              xmax=float(record[5]),
                              ymax=float(record[6]),
                              x_pos=int(record[8]),
                              y_pos=int(record[9]))
        return bbox

    @staticmethod
    def _parse_detections(data, record_size, converter):
        assert data.size % record_size == 0, 'incorrect record_size'
        records = data.reshape([-1, record_size])

        detections = []
        for record in records:
            detection = converter(record)
            detections.append(detection)

        return detections

    @staticmethod
    def _match_detections_with_actions(detections, anchors):
        actions = []
        for det in detections:
            action_prediction =\
                anchors[det.anchor_id][det.item_id, det.y_pos, det.x_pos, :]

            action_label = np.argmax(action_prediction)
            action_conf = np.max(action_prediction)

            action = OutputDetection(item_id=det.item_id,
                                     action=action_label,
                                     det_conf=det.det_conf,
                                     action_conf=action_conf,
                                     xmin=det.xmin, ymin=det.ymin,
                                     xmax=det.xmax, ymax=det.ymax,
                                     anchor_id=det.anchor_id,
                                     x_pos=det.x_pos, y_pos=det.y_pos)
            actions.append(action)
        return actions

    @staticmethod
    def _convert_actions_to_blob(actions, record_size):
        records = []
        for action in actions:
            record = [action.item_id, action.action,
                      action.det_conf, action.action_conf,
                      action.xmin, action.ymin, action.xmax, action.ymax,
                      action.anchor_id, action.x_pos, action.y_pos]
            assert len(record) == record_size

            records.append(record)

        out_blob = np.array(records, dtype=np.float32)
        out_blob = out_blob.reshape([1, 1, len(records), record_size])

        return out_blob

    def _load_params(self, param_str):
        layer_params = eval(param_str)

        assert 'num_anchors' in layer_params
        self._num_anchors = layer_params['num_anchors']

    def _init_states(self):
        pass

    def setup(self, bottom, top):
        self._load_params(self.param_str)
        self._init_states()

    def forward(self, bottom, top):
        try:
            assert len(bottom) == self._num_anchors + 1
            assert len(top) == 1

            detections_data = np.array(bottom[0].data)

            anchors_data = []
            for i in xrange(self._num_anchors):
                anchors_data.append(np.array(bottom[i + 1].data))

            all_detections = self._parse_detections(
                detections_data, INPUT_RECORD_SIZE, self._translate_prediction)

            all_actions = self._match_detections_with_actions(all_detections, anchors_data)

            matches_blob = self._convert_actions_to_blob(all_actions, OUTPUT_RECORD_SIZE)
            out_shape = matches_blob.shape

            top[0].reshape(out_shape[0], out_shape[1], out_shape[2], out_shape[3])
            top[0].data[...] = matches_blob
        except Exception:
            LOG('ExtendedActionsDetectionOutputLayer exception: {}'.format(traceback.format_exc()))
            exit()

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(1, 1, 1, OUTPUT_RECORD_SIZE)


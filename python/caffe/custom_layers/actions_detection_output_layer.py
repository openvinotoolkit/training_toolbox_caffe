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

import traceback
from collections import namedtuple

import numpy as np

from caffe._caffe import log as LOG
from caffe._caffe import Layer as BaseLayer


InputDetection = namedtuple('InputDetection', 'item_id, det_conf, anchor_id,'
                                              'xmin, ymin, xmax, ymax,'
                                              'x_pos, y_pos')
OutputDetection = namedtuple('OutputDetection', 'item_id, action,'
                                                'det_conf, action_conf,'
                                                'xmin, ymin, xmax, ymax')

INPUT_RECORD_SIZE = 10
OUTPUT_RECORD_SIZE = 8


class ActionsDetectionOutputLayer(BaseLayer):
    """Python-based layer to output detections in SSD format.
       New detection output format includes 3 additional fields:
           * anchor branch ID
           * x coordinate of bbox cell on feature map
           * y coordinate of bbox cell on feature map
    """

    @staticmethod
    def _translate_prediction(record):
        """Decodes the input record into the human-readable format.

        :param record: Input single record for decoding
        :return: Human-readable record
        """

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
        """ Convert input blob into list of human-readable records.

        :param data: Input blob
        :param record_size: Size of each input record
        :param converter: Function to convert input record
        :return: List of detections
        """

        assert data.size % record_size == 0, 'incorrect record_size'
        records = data.reshape([-1, record_size])

        detections = []
        for record in records:
            detection = converter(record)
            detections.append(detection)

        return detections

    @staticmethod
    def _match_detections_with_actions(detections, anchors):
        """Carry out matching of detections with appropriate action classes.

        Extracted class for each detection is determined as max confidential response
        from appropriate position of anchor branch.

        :param detections: List of input detections
        :param anchors: Anchor branches
        :return: Same list of detections but with action class info
        """

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
                                     xmin=det.xmin,
                                     ymin=det.ymin,
                                     xmax=det.xmax,
                                     ymax=det.ymax)
            actions.append(action)
        return actions

    @staticmethod
    def _convert_actions_to_blob(actions, record_size):
        """ Carry out back conversation of list of detections to output blob.

        :param actions: List of records for conversation
        :param record_size: Size of output record
        :return: Blob with annotation
        """

        records = []
        for action in actions:
            record = [action.item_id, action.action,
                      action.det_conf, action.action_conf,
                      action.xmin, action.ymin, action.xmax, action.ymax]
            assert len(record) == record_size

            records.append(record)

        out_blob = np.array(records, dtype=np.float32)
        out_blob = out_blob.reshape([1, 1, len(records), record_size])

        return out_blob

    @staticmethod
    def _complex_nms(actions, min_overlap, num_actions):
        """ Carry out extended variant of NMS algorithm to count number of votes
            for the final action class.

        :param actions: List of detections
        :param min_overlap: Min overlap (IoU metric) to merge bboxes
        :param num_actions: Total number of valid actions
        :return: Final list of detections
        """

        def _iou(box_a, box_b):
            """ Calculates Intersection over Union (IoU) metric.

            :param box_a: First bbox
            :param box_b: Second bbox
            :return: Scalar value of metric
            """

            top_left_x = max(box_a.xmin, box_b.xmin)
            top_left_y = max(box_a.ymin, box_b.ymin)
            intersect_width = max(0.0, min(box_a.xmax, box_b.xmax) - top_left_x)
            intersect_height = max(0.0, min(box_a.ymax, box_b.ymax) - top_left_y)
            intersection_area = float(intersect_width * intersect_height)

            area1 = (box_a.xmax - box_a.xmin) * (box_a.ymax - box_a.ymin)
            area2 = (box_b.xmax - box_b.xmin) * (box_b.ymax - box_b.ymin)

            union_area = float(area1 + area2 - intersection_area)

            return intersection_area / union_area if union_area > 0.0 else 0.0

        indexed_scores = [(action.det_conf, i) for i, action in enumerate(actions)]
        indexed_scores.sort(key=lambda t: t[0], reverse=True)

        out_indices = []
        for _, anchor_idx in indexed_scores:
            anchor_det = actions[anchor_idx]

            if len(out_indices) == 0:
                class_votes = np.zeros([num_actions], dtype=np.float32)
                class_votes[anchor_det.action] = 1.0
                out_indices.append([anchor_idx, class_votes])
                continue

            overlaps = np.array([_iou(anchor_det, actions[ref_idx]) for ref_idx, _ in out_indices])

            max_overlap = np.max(overlaps)
            if max_overlap > min_overlap:
                argmax_overlap = np.argmax(overlaps)
                out_indices[argmax_overlap][1][anchor_det.action] += anchor_det.action_conf
            else:
                class_votes = np.zeros([num_actions], dtype=np.float32)
                class_votes[anchor_det.action] = anchor_det.action_conf
                out_indices.append([anchor_idx, class_votes])

        out_actions = [OutputDetection(item_id=actions[i].item_id,
                                       action=np.argmax(cl),
                                       det_conf=actions[i].det_conf,
                                       action_conf=actions[i].action_conf,
                                       xmin=actions[i].xmin, ymin=actions[i].ymin,
                                       xmax=actions[i].xmax, ymax=actions[i].ymax) for i, cl in out_indices]

        return out_actions

    def _load_params(self, param_str):
        """Loads layer parameters.

        :param param_str: Input str of parameters
        """

        layer_params = eval(param_str)

        assert 'num_anchors' in layer_params
        self._num_anchors = layer_params['num_anchors']

        self._use_nms = layer_params['use_nms'] if 'use_nms' in layer_params else False
        if self._use_nms:
            assert 'num_valid_actions' in layer_params
            self._num_valid_actions = layer_params['num_valid_actions']

            self._min_overlap = layer_params['min_overlap'] if 'min_overlap' in layer_params else 0.45

    def setup(self, bottom, top):
        """Initializes layer.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        self._load_params(self.param_str)

    def forward(self, bottom, top):
        """Carry out forward pass.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

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

            if self._use_nms:
                all_actions = self._complex_nms(all_actions, self._min_overlap, self._num_valid_actions)

            matches_blob = self._convert_actions_to_blob(all_actions, OUTPUT_RECORD_SIZE)
            out_shape = matches_blob.shape

            top[0].reshape(out_shape[0], out_shape[1], out_shape[2], out_shape[3])
            top[0].data[...] = matches_blob
        except Exception:
            LOG('MatcherLayer exception: {}'.format(traceback.format_exc()))
            exit()

    def backward(self, top, propagate_down, bottom):
        """Carry out backward pass.

        :param top: List of top blobs
        :param propagate_down: List of indicators to carry out back-propagation for
                               the specified bottom blob
        :param bottom: List of bottom blobs
        """

        pass

    def reshape(self, bottom, top):
        """Carry out blob reshaping.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        top[0].reshape(1, 1, 1, OUTPUT_RECORD_SIZE)

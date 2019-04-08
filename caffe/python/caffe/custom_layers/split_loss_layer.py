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
from builtins import range
from collections import namedtuple

import numpy as np

from caffe._caffe import log as LOG
from caffe._caffe import Layer as BaseLayer


BBoxDesc = namedtuple('BBoxDesc', 'item, det_conf, anchor, action, xmin, ymin, xmax, ymax, x, y')

MATCHED_RECORD_SIZE = 11


class SplitLossLayer(BaseLayer):
    """One of Metric-learning losses which forces incorrect predicted samples from
       different classes and from neighboring cells to be far from each other.

       Current implementation is able to extract embeddings from the anchor branches
       by the specified list of detections.
    """

    @staticmethod
    def _translate_matched_prediction(record):
        """Decodes the input record into the human-readable format.

        :param record: Input single record for decoding
        :return: Human-readable record
        """

        bbox = BBoxDesc(item=int(record[0]),
                        det_conf=float(record[1]),
                        anchor=int(record[6]),
                        action=int(record[8]),
                        xmin=float(record[2]),
                        ymin=float(record[3]),
                        xmax=float(record[4]),
                        ymax=float(record[5]),
                        x=int(record[9]),
                        y=int(record[10]))
        return bbox

    @staticmethod
    def _read_detections(data, record_size, converter, valid_action_ids, min_conf):
        """Convert input blob into list of human-readable records.

        :param data: Input blob
        :param record_size: Size of each input record
        :param converter: Function to convert input record
        :param valid_action_ids:
        :param min_conf: List of IDs of valid actions
        :return: List of detections
        """

        assert data.size % record_size == 0, 'incorrect record_size'
        records = data.reshape([-1, record_size])

        detections = {}
        for record in records:
            detection = converter(record)

            if detection.det_conf < min_conf or detection.item < 0:
                continue

            if valid_action_ids is None or detection.action in valid_action_ids:
                detections[detection.item] = detections.get(detection.item, []) + [detection]

        return detections

    @staticmethod
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

    def _load_params(self, param_str):
        """Loads layer parameters.

        :param param_str: Input str of parameters
        """

        layer_params = eval(param_str)

        assert 'num_anchors' in layer_params

        self._num_anchors = layer_params['num_anchors']

        self._margin = float(layer_params['margin']) if 'margin' in layer_params else 0.6
        self._min_overlap = float(layer_params['min_overlap']) if 'min_overlap' in layer_params else 0.3
        self._min_conf = float(layer_params['min_conf']) if 'min_conf' in layer_params else 0.01
        self._valid_action_ids = layer_params['valid_action_ids'] if 'valid_action_ids' in layer_params else None

        self._candidates = []
        self._embeddings = []

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
            assert len(bottom) == self._num_anchors + 2
            assert len(top) == 1 or len(top) == 4

            detections_data = np.array(bottom[0].data)
            batch_detections = self._read_detections(detections_data, MATCHED_RECORD_SIZE,
                                                     self._translate_matched_prediction,
                                                     self._valid_action_ids, self._min_conf)

            centers = np.array(bottom[1].data)

            self._embeddings = []
            for i in range(self._num_anchors):
                self._embeddings.append(np.array(bottom[i + 2].data))

            all_candidates = []
            total_num_pairs = 0
            total_num_overlapped = 0
            total_num_incorrect = 0
            for item_id in batch_detections:
                detections = batch_detections[item_id]

                for i, _ in enumerate(detections):
                    anchor_det = detections[i]
                    for j in range(i + 1, len(detections)):
                        ref_det = detections[j]

                        # exclude same class predictions
                        if anchor_det.action == ref_det.action:
                            continue

                        overlap = self._iou(anchor_det, ref_det)
                        if overlap < self._min_overlap:
                            continue

                        total_num_overlapped += 1

                        anchor_embed = self._embeddings[anchor_det.anchor][anchor_det.item, :,
                                                                           anchor_det.y, anchor_det.x]
                        ref_embed = self._embeddings[ref_det.anchor][ref_det.item, :, ref_det.y, ref_det.x]

                        anchor_center_distances =\
                            (1.0 - np.matmul(centers, anchor_embed.reshape([-1, 1]))).reshape([-1])
                        ref_center_distances =\
                            (1.0 - np.matmul(centers, ref_embed.reshape([-1, 1]))).reshape([-1])

                        anchor_action = np.argmin(anchor_center_distances)
                        ref_action = np.argmin(ref_center_distances)

                        # exclude well-separated predictions
                        if anchor_action != ref_action:
                            continue

                        # exclude predictions with both incorrect labels
                        if anchor_action != anchor_det.action and ref_action != ref_det.action:
                            continue

                        total_num_incorrect += 1

                        embed_dist = 1.0 - np.sum(anchor_embed * ref_embed)

                        if anchor_action != anchor_det.action:
                            loss = self._margin + anchor_center_distances[anchor_det.action] - embed_dist
                            if loss > 0.0:
                                all_candidates.append((loss, embed_dist, anchor_det, ref_det))
                            total_num_pairs += 1

                        if ref_action != ref_det.action:
                            loss = self._margin + ref_center_distances[ref_det.action] - embed_dist
                            if loss > 0.0:
                                all_candidates.append((loss, embed_dist, ref_det, anchor_det))
                            total_num_pairs += 1

            if len(all_candidates) == 0:
                self._candidates = []

                top[0].data[...] = 0.0
                if len(top) == 4:
                    top[1].data[...] = 0.0
                    top[2].data[...] = 0.0
                    top[3].data[...] = 0.0
            else:
                self._candidates = all_candidates

                loss = np.sum([tup[0] for tup in self._candidates]) / float(len(self._candidates))
                top[0].data[...] = loss

                if len(top) == 4:
                    top[1].data[...] = np.median([tup[1] for tup in self._candidates])
                    top[2].data[...] = float(len(self._candidates)) / float(total_num_pairs)
                    top[3].data[...] = float(total_num_incorrect) / float(total_num_overlapped)
        except Exception:
            LOG('SplitLossLayer forward pass exception: {}'.format(traceback.format_exc()))
            exit()

    def backward(self, top, propagate_down, bottom):
        """Carry out backward pass.

        :param top: List of top blobs
        :param propagate_down: List of indicators to carry out back-propagation for
                               the specified bottom blob
        :param bottom: List of bottom blobs
        """

        try:
            if propagate_down[0]:
                raise Exception('Cannot propagate down through the matched detections')

            centers_data = np.array(bottom[1].data)
            centers_diff_data = np.zeros(bottom[1].data.shape) if propagate_down[1] else None

            diff_data = {}
            for anchor_id in range(self._num_anchors):
                if propagate_down[anchor_id + 2]:
                    diff_data[anchor_id] = np.zeros(bottom[anchor_id + 2].data.shape)

            if len(self._candidates) > 0:
                factor = top[0].diff[0] / float(len(self._candidates))
                for _, _, anchor_det, ref_det in self._candidates:
                    anchor_embed = self._embeddings[anchor_det.anchor][anchor_det.item, :, anchor_det.y, anchor_det.x]
                    ref_embedding = self._embeddings[ref_det.anchor][ref_det.item, :, ref_det.y, ref_det.x]

                    if propagate_down[anchor_det.anchor + 2]:
                        diff_data[anchor_det.anchor][anchor_det.item, :, anchor_det.y, anchor_det.x] \
                            += factor * (ref_embedding - centers_data[anchor_det.action])

                    if propagate_down[ref_det.anchor + 2]:
                        diff_data[ref_det.anchor][ref_det.item, :, ref_det.y, ref_det.x] += factor * anchor_embed

                    if centers_diff_data is not None:
                        centers_diff_data[anchor_det.action] += -factor * anchor_embed

            if centers_diff_data is not None:
                bottom[1].diff[...] = centers_diff_data

            for anchor_id in range(self._num_anchors):
                if propagate_down[anchor_id + 2]:
                    bottom[anchor_id + 2].diff[...] = diff_data[anchor_id]
        except Exception:
            LOG('SplitLossLayer backward pass exception: {}'.format(traceback.format_exc()))
            exit()

    def reshape(self, bottom, top):
        """Carry out blob reshaping.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        top[0].reshape(1)

        if len(top) == 4:
            top[1].reshape(1)
            top[2].reshape(1)
            top[3].reshape(1)

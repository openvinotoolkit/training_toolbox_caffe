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


BBoxDesc = namedtuple('BBoxDesc', 'item, det_conf, anchor, action, x, y')

MATCHED_RECORD_SIZE = 11


class GlobPushLossLayer(BaseLayer):
    """One of Metric-learning losses which forces valid samples of each class
       to be far from the background samples.

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
    def _outer_class_mask(detections, height, width):
        """Estimates mask of invalid (unmatched with ground-truth detections) pixels.

        :param detections: List of matched with gt detections
        :param height: Height of target mask
        :param width: Width of target mask
        :return: mask of invalid pixels
        """

        mask = np.ones([height, width], dtype=np.bool)
        for det in detections:
            mask[det.y, det.x] = False
        return mask

    def _load_params(self, param_str):
        """Loads layer parameters.

        :param param_str: Input str of parameters
        """

        layer_params = eval(param_str)

        assert 'num_anchors' in layer_params

        self._num_anchors = layer_params['num_anchors']

        self._margin = float(layer_params['margin']) if 'margin' in layer_params else 1.0
        self._min_conf = float(layer_params['min_conf']) if 'min_conf' in layer_params else 0.01
        self._valid_action_ids = layer_params['valid_action_ids'] if 'valid_action_ids' in layer_params else None

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
            assert len(top) == 1 or len(top) == 3

            detections_data = np.array(bottom[0].data)
            batch_detections = self._read_detections(detections_data, MATCHED_RECORD_SIZE,
                                                     self._translate_matched_prediction,
                                                     self._valid_action_ids, self._min_conf)

            centers_data = np.array(bottom[1].data)
            num_centers = centers_data.shape[0]

            self._embeddings = []
            for i in range(self._num_anchors):
                self._embeddings.append(np.array(bottom[i + 2].data))

            height, width = self._embeddings[0].shape[2:]

            all_masks = []
            out_loss = 0.0
            total_num_pairs = 0
            valid_num_pairs = 0
            dist_sum = 0.0
            for item_id in batch_detections:
                detections = batch_detections[item_id]

                outer_mask = self._outer_class_mask(detections, height, width)
                total_num_pairs += self._num_anchors * num_centers * int(np.sum(outer_mask))

                anchor_masks = []
                for anchor_id in range(self._num_anchors):
                    anchor_embeddings = self._embeddings[anchor_id][item_id]

                    center_masks = []
                    for center_id in range(num_centers):
                        center_embedding = centers_data[center_id].reshape([-1, 1, 1])

                        distances = 1.0 - np.sum(anchor_embeddings * center_embedding, axis=0)
                        losses = self._margin - distances

                        out_mask = (losses > 0.0) * outer_mask
                        valid_num_pairs += int(np.sum(out_mask))
                        out_loss += np.sum(losses[out_mask])
                        dist_sum += np.sum(distances[out_mask])

                        center_masks.append(out_mask)

                    anchor_masks.append(center_masks)

                all_masks.append(anchor_masks)

            top[0].data[...] = out_loss / float(valid_num_pairs) if valid_num_pairs > 0 else 0.0
            if len(top) == 3:
                top[1].data[...] = float(valid_num_pairs) / float(total_num_pairs) if total_num_pairs > 0 else 0.0
                top[2].data[...] = dist_sum / float(valid_num_pairs) if valid_num_pairs > 0 else 0.0

            self._valid_num_pairs = valid_num_pairs
            self._masks = all_masks
        except Exception:
            LOG('GlobPushLossLayer forward pass exception: {}'.format(traceback.format_exc()))
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

            anchor_diff_data = {}
            for anchor_id in range(self._num_anchors):
                if propagate_down[anchor_id + 2]:
                    anchor_diff_data[anchor_id] = np.zeros(bottom[anchor_id + 2].data.shape)

            if self._valid_num_pairs > 0:
                factor = top[0].diff[0] / float(self._valid_num_pairs)
                for item_id, _ in enumerate(self._masks):
                    anchor_masks = self._masks[item_id]
                    for anchor_id, _ in enumerate(anchor_masks):
                        embeddings = self._embeddings[anchor_id][item_id]
                        diff_data = anchor_diff_data[anchor_id][item_id]

                        embedding_size = embeddings.shape[0]

                        center_masks = anchor_masks[anchor_id]
                        for center_id, _ in enumerate(center_masks):
                            mask = center_masks[center_id]
                            num_pairs = int(np.sum(mask))
                            mask = np.tile(np.expand_dims(mask, axis=0), reps=[embedding_size, 1, 1])

                            if centers_diff_data is not None:
                                filtered_embeddings = embeddings[mask].reshape([embedding_size, -1])
                                centers_diff_data[center_id] += factor * np.sum(filtered_embeddings, axis=1)

                            if propagate_down[anchor_id + 2]:
                                diff_data[mask] += factor * np.tile(centers_data[center_id].reshape([-1, 1]),
                                                                    reps=[1, num_pairs]).reshape([-1])

            if centers_diff_data is not None:
                bottom[1].diff[...] = centers_diff_data

            for anchor_id in range(self._num_anchors):
                if propagate_down[anchor_id + 2]:
                    bottom[anchor_id + 2].diff[...] = anchor_diff_data[anchor_id]
        except Exception:
            LOG('GlobPushLossLayer backward pass exception: {}'.format(traceback.format_exc()))
            exit()

    def reshape(self, bottom, top):
        """Carry out blob reshaping.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        top[0].reshape(1)

        if len(top) == 3:
            top[1].reshape(1)
            top[2].reshape(1)

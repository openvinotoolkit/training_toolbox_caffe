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


BBoxDesc = namedtuple('BBoxDesc', 'item, det_conf, instance_id, anchor, action, x, y')

MATCHED_RECORD_SIZE = 11


class PushLossLayer(BaseLayer):
    """One of Metric-learning losses which forces valid samples from different classes
       to be far from each other.

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
                        instance_id=int(record[7]),
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

        detections = {i: [] for i, _ in enumerate(valid_action_ids)}
        for record in records:
            detection = converter(record)

            if detection.det_conf < min_conf or detection.item < 0:
                continue

            if detection.action in valid_action_ids:
                detections[detection.action].append(detection)

        return detections

    @staticmethod
    def _filter_detections(all_detections, max_num_samples):
        """Generates max_num_samples of samples for each valid class.

        :param all_detections: List of all detected bboxes
        :param max_num_samples: Maximal number of samples per class
        :return: List samples splitted by class
        """

        out_detections = {}
        for class_id in all_detections:
            in_class_detections = all_detections[class_id]

            instances = {}
            for det in in_class_detections:
                if det.instance_id not in instances:
                    instances[det.instance_id] = []
                instances[det.instance_id].append(det)

            if len(instances) == 0:
                continue

            detection_candidates = []
            num_samples_per_instance = int(np.ceil(float(max_num_samples) / float(len(instances))))
            for instance_id in instances:
                instance_detections = instances[instance_id]

                max_num_rand_samples = min(len(instance_detections), num_samples_per_instance)
                rand_indices = np.random.choice(range(len(instance_detections)), max_num_rand_samples,
                                                replace=False)
                for i in rand_indices:
                    detection_candidates.append(instance_detections[i])

            if len(detection_candidates) == 0:
                continue
            elif len(detection_candidates) <= max_num_samples:
                out_detections[class_id] = detection_candidates
            else:
                rand_indices = np.random.choice(range(len(detection_candidates)), max_num_samples,
                                                replace=False)
                out_detections[class_id] = [detection_candidates[i] for i in rand_indices]

        return out_detections

    def _load_params(self, param_str):
        """Loads layer parameters.

        :param param_str: Input str of parameters
        """

        layer_params = eval(param_str)

        assert 'num_anchors' in layer_params
        assert 'valid_action_ids' in layer_params

        self._num_anchors = layer_params['num_anchors']
        self._valid_action_ids = layer_params['valid_action_ids']
        assert len(self._valid_action_ids) > 0

        self._margin = float(layer_params['margin']) if 'margin' in layer_params else 1.0
        self._min_conf = float(layer_params['min_conf']) if 'min_conf' in layer_params else 0.01
        assert self._min_conf >= 0.0
        self._max_num_samples = layer_params['max_num_samples'] if 'max_num_samples' in layer_params else 10

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
            assert len(top) == 1 or len(top) == 3

            detections_data = np.array(bottom[0].data)
            all_detections = self._read_detections(detections_data, MATCHED_RECORD_SIZE,
                                                   self._translate_matched_prediction,
                                                   self._valid_action_ids, self._min_conf)
            detections = self._filter_detections(all_detections, self._max_num_samples)

            classes = list(detections)
            class_pairs = [(classes[i], classes[j]) for i, _ in enumerate(classes) for j in range(i + 1, len(classes))]

            self._embeddings = []
            for i in range(self._num_anchors):
                self._embeddings.append(np.array(bottom[i + 1].data))

            all_candidates = []
            total_num_pairs = 0
            for class_i, class_j in class_pairs:
                detections_i = detections[class_i]
                detections_j = detections[class_j]
                if len(detections_i) == 0 or len(detections_j) == 0:
                    continue

                for i, _ in enumerate(detections_i):
                    anchor_det = detections_i[i]
                    anchor_embed = self._embeddings[anchor_det.anchor][anchor_det.item, :,
                                                                       anchor_det.y, anchor_det.x]

                    for j, _ in enumerate(detections_j):
                        ref_det = detections_j[j]
                        ref_embed = self._embeddings[ref_det.anchor][ref_det.item, :, ref_det.y, ref_det.x]

                        embed_dist = 1.0 - np.sum(anchor_embed * ref_embed)
                        loss = self._margin - embed_dist
                        if loss > 0.0:
                            all_candidates.append((loss, embed_dist, anchor_det, ref_det))

                        total_num_pairs += 1

            if len(all_candidates) == 0:
                self._candidates = []

                top[0].data[...] = 0.0
                if len(top) == 3:
                    top[1].data[...] = 0.0
                    top[2].data[...] = 0.0
            else:
                if len(all_candidates) > 2:
                    threshold_value = np.median([tup[0] for tup in all_candidates])
                    self._candidates = [tup for tup in all_candidates if tup[0] > threshold_value]
                else:
                    self._candidates = all_candidates

                loss = np.sum([tup[0] for tup in self._candidates]) / float(len(self._candidates))
                top[0].data[...] = loss

                if len(top) == 3:
                    top[1].data[...] = np.median([tup[1] for tup in self._candidates])
                    top[2].data[...] = float(len(all_candidates)) / float(total_num_pairs)
        except Exception:
            LOG('PushLossLayer forward pass exception: {}'.format(traceback.format_exc()))
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

            diff_data = {}
            for anchor_id in range(self._num_anchors):
                if propagate_down[anchor_id + 1]:
                    diff_data[anchor_id] = np.zeros(bottom[anchor_id + 1].data.shape)

            factor = top[0].diff[0] / float(len(self._candidates)) if len(self._candidates) > 0 else 0.0
            for _, _, anchor_det, ref_det in self._candidates:
                if propagate_down[anchor_det.anchor + 1]:
                    diff_data[anchor_det.anchor][anchor_det.item, :, anchor_det.y, anchor_det.x] \
                        += factor * self._embeddings[ref_det.anchor][ref_det.item, :, ref_det.y, ref_det.x]

                if propagate_down[ref_det.anchor + 1]:
                    diff_data[ref_det.anchor][ref_det.item, :, ref_det.y, ref_det.x] \
                        += factor * self._embeddings[anchor_det.anchor][anchor_det.item, :, anchor_det.y, anchor_det.x]

            for anchor_id in range(self._num_anchors):
                if propagate_down[anchor_id + 1]:
                    bottom[anchor_id + 1].diff[...] = diff_data[anchor_id]
        except Exception:
            LOG('PushLossLayer backward pass exception: {}'.format(traceback.format_exc()))
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

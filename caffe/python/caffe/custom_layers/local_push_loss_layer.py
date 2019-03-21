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
from six import itervalues

from caffe._caffe import log as LOG
from caffe._caffe import Layer as BaseLayer


BBoxDesc = namedtuple('BBoxDesc', 'item, id, det_conf, anchor, action, x, y')

MATCHED_RECORD_SIZE = 11


class LocalPushLossLayer(BaseLayer):
    """One of Metric-learning losses which forces valid samples of each class
       to be far from the centers of other classes.

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
                        id=int(record[7]),
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

        detections = []
        for record in records:
            detection = converter(record)

            if detection.det_conf < min_conf or detection.item < 0:
                continue

            if valid_action_ids is None or detection.action in valid_action_ids:
                detections.append(detection)

        return detections

    @staticmethod
    def _inner_masks(detections, height, width, num_classes):
        """Estimates mask of valid (matched with ground-truth detections) pixels.

        :param detections: List of matched with gt detections
        :param height: Height of target mask
        :param width: Width of target mask
        :param num_classes: Target number of valid classes
        :return: mask of valid pixels
        """

        masks = [np.zeros([height, width], dtype=np.bool) for _ in range(num_classes)]
        for det in detections:
            masks[det.action][det.y, det.x] = True
        return masks

    @staticmethod
    def _estimate_weights(new_frequencies, smoothed_frequencies, gamma, limits):
        for cl_id in new_frequencies:
            if smoothed_frequencies[cl_id] > 0.0:
                smoothed_frequencies[cl_id] = \
                    gamma * smoothed_frequencies[cl_id] + (1.0 - gamma) * new_frequencies[cl_id]
            else:
                smoothed_frequencies[cl_id] = new_frequencies[cl_id]

        sum_smoothed_values = np.sum([val for val in itervalues(smoothed_frequencies) if val > 0.0])
        normalizer = sum_smoothed_values / float(len(smoothed_frequencies))

        new_weights = {}
        for cl_id in smoothed_frequencies:
            new_weight = normalizer / smoothed_frequencies[cl_id] if smoothed_frequencies[cl_id] > 0.0 else 0.0
            if limits[0] is not None and new_weight < limits[0]:
                new_weight = limits[0]
            elif limits[1] is not None and new_weight > limits[1]:
                new_weight = limits[1]

            new_weights[cl_id] = new_weight

        return new_weights

    def _load_params(self, param_str):
        """Loads layer parameters.

        :param param_str: Input str of parameters
        """

        layer_params = eval(param_str)

        assert 'num_anchors' in layer_params
        assert 'valid_action_ids' in layer_params

        self._num_anchors = layer_params['num_anchors']
        assert self._num_anchors > 0
        self._valid_action_ids = layer_params['valid_action_ids']
        assert len(self._valid_action_ids) > 0

        self._margin = float(layer_params['margin']) if 'margin' in layer_params else 0.6
        assert self._margin >= 0.0
        self._min_conf = float(layer_params['min_conf']) if 'min_conf' in layer_params else 0.01
        assert self._min_conf >= 0.0
        self._adaptive_weights = layer_params['adaptive_weights'] if 'adaptive_weights' in layer_params else False
        if self._adaptive_weights:
            self._gamma = float(layer_params['gamma']) if 'gamma' in layer_params else 0.9
            min_weight = float(layer_params['min_weight']) if 'min_weight' in layer_params else None
            max_weight = float(layer_params['max_weight']) if 'max_weight' in layer_params else None
            assert min_weight < max_weight
            self._weight_limits = [min_weight, max_weight]
        else:
            self._class_weights = layer_params['weights'] if 'weights' in layer_params else None
            if self._class_weights is None:
                self._class_weights = np.ones([len(self._valid_action_ids)], dtype=np.float32)
            else:
                assert np.all([w > 0.0 for w in self._class_weights])
        self._instance_norm = layer_params['instance_norm'] if 'instance_norm' in layer_params else False

    def _init_states(self):
        """Initializes internal state"""

        if self._adaptive_weights:
            self._smoothed_frequencies = {cl_id: -1.0 for cl_id in self._valid_action_ids}

    def setup(self, bottom, top):
        """Initializes layer.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        self._load_params(self.param_str)
        self._init_states()

    def forward(self, bottom, top):
        """Carry out forward pass.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        try:
            assert len(bottom) == self._num_anchors + 2
            assert len(top) == 1 or len(top) == 4

            detections_data = np.array(bottom[0].data)
            all_detections = self._read_detections(detections_data, MATCHED_RECORD_SIZE,
                                                   self._translate_matched_prediction,
                                                   self._valid_action_ids, self._min_conf)

            centers_data = np.array(bottom[1].data)
            num_centers = centers_data.shape[0]

            self._embeddings = []
            for i in range(self._num_anchors):
                self._embeddings.append(np.array(bottom[i + 2].data))

            if self._adaptive_weights:
                total_num = 0
                class_counts = {cl_id: 0 for cl_id in self._valid_action_ids}
                for det in all_detections:
                    class_counts[det.action] += 1
                    total_num += 1

                normalizer = 1.0 / float(total_num) if total_num > 0 else 0.0
                class_frequencies = {cl_id: normalizer * class_counts[cl_id] for cl_id in self._valid_action_ids}

                class_weights = self._estimate_weights(class_frequencies, self._smoothed_frequencies,
                                                       self._gamma, self._weight_limits)
            else:
                class_weights = self._class_weights

            pos_matches = []
            total_num_matches = 0
            sum_pos_distances = 0.0
            sum_neg_distances = 0.0
            losses = []
            instance_counts = {}
            instance_weights = []

            for det in all_detections:
                det_embedding = self._embeddings[det.anchor][det.item, :, det.y, det.x]
                center_embedding = centers_data[det.action]

                pos_distance = 1.0 - np.sum(det_embedding * center_embedding)

                for center_id in range(num_centers):
                    if center_id == det.action:
                        continue

                    neg_distance = 1.0 - np.sum(det_embedding * centers_data[center_id])

                    loss = self._margin + pos_distance - neg_distance
                    if loss > 0.0:
                        losses.append(loss)

                        sum_pos_distances += pos_distance
                        sum_neg_distances += neg_distance

                        pos_matches.append((det, center_id))
                        instance_weights.append(class_weights[det.action])

                        if det.item not in instance_counts:
                            instance_counts[det.item] = {}
                        local_instance_counts = instance_counts[det.item]
                        if det.id not in local_instance_counts:
                            local_instance_counts[det.id] = 0
                        local_instance_counts[det.id] += 1

                    total_num_matches += 1

            if self._instance_norm:
                instance_weights = \
                    [instance_weights[i] / float(instance_counts[pos_matches[i][0].item][pos_matches[i][0].id])
                     for i in range(len(pos_matches))]
                num_instances = np.sum([len(counts) for counts in itervalues(instance_counts)])
            else:
                instance_weights = [instance_weights[i] for i, _ in enumerate(pos_matches)]
                num_instances = len(pos_matches)

            weighted_sum_losses = np.sum([instance_weights[i] * losses[i] for i, _ in enumerate(pos_matches)])

            top[0].data[...] = weighted_sum_losses / float(num_instances) if num_instances > 0 else 0.0
            if len(top) == 4:
                top[1].data[...] = float(len(pos_matches)) / float(total_num_matches) if total_num_matches > 0 else 0.0
                top[2].data[...] = float(sum_pos_distances) / float(len(pos_matches)) if len(pos_matches) > 0 else 0.0
                top[3].data[...] = float(sum_neg_distances) / float(len(pos_matches)) if len(pos_matches) > 0 else 0.0

            self._pos_matches = pos_matches
            self._weights = instance_weights
            self._num_instances = num_instances
        except Exception:
            LOG('LocalPushLossLayer forward pass exception: {}'.format(traceback.format_exc()))
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

            if len(self._pos_matches) > 0:
                factor = top[0].diff[0] / float(self._num_instances)

                for i, _ in enumerate(self._pos_matches):
                    det, center_id = self._pos_matches[i]
                    loss_weight = self._weights[i]

                    if propagate_down[det.anchor + 2]:
                        anchor_diff_data[det.anchor][det.item, :, det.y, det.x]\
                            += factor * loss_weight * (centers_data[center_id] - centers_data[det.action])

                    if centers_diff_data is not None:
                        embedding = self._embeddings[det.anchor][det.item, :, det.y, det.x]

                        centers_diff_data[det.action] += -factor * loss_weight * embedding
                        centers_diff_data[center_id] += factor * loss_weight * embedding

            if centers_diff_data is not None:
                bottom[1].diff[...] = centers_diff_data

            for anchor_id in range(self._num_anchors):
                if propagate_down[anchor_id + 2]:
                    bottom[anchor_id + 2].diff[...] = anchor_diff_data[anchor_id]
        except Exception:
            LOG('LocalPushLossLayer backward pass exception: {}'.format(traceback.format_exc()))
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

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


class CenterLossLayer(BaseLayer):
    """One of Metric-learning losses which forces samples from the same class
       to be close to their center.

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
    def _get_scale(step, start, end, num_steps, power, use_last_scale):
        """Returns scalar which value is polynomially depends of iteration number.

        :param step: Current iteration number
        :param start: Initial scalar value
        :param end: Final scalar value
        :param num_steps: Total number of steps
        :param power: Power of polynomial
        :param use_last_scale: Whether to use end value instead of counting
        :return: Current scalar value
        """

        if use_last_scale:
            out_scale = end
        else:
            factor = float(end - start) / float(1 - power)
            var_a = factor / (float(num_steps) ** float(power))
            var_b = -factor * float(power) / float(num_steps)
            scale =\
                var_a * float(step) ** float(power) + var_b * float(step) + float(start) if step < num_steps else end
            out_scale = scale

        return out_scale

    @staticmethod
    def _estimate_weights(new_frequencies, smoothed_frequencies, gamma, limits):
        for cl_id in new_frequencies:
            if smoothed_frequencies[cl_id] > 0.0:
                smoothed_frequencies[cl_id] =\
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

        self._use_filtering = layer_params['use_filtering'] if 'use_filtering' in layer_params else False
        self._entropy_weight = float(layer_params['entropy_weight']) if 'entropy_weight' in layer_params else 0.4
        self._init_scale = layer_params['init_scale'] if 'init_scale' in layer_params else 16
        self._target_scale = layer_params['target_scale'] if 'target_scale' in layer_params else 1.5
        self._num_steps = layer_params['num_steps'] if 'num_steps' in layer_params else 140000
        self._power = layer_params['power'] if 'power' in layer_params else 2.0
        self._use_last = layer_params['use_last'] if 'use_last' in layer_params else False

    def _init_states(self):
        """Initializes internal state"""

        self._step = 0

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
            assert len(top) == 1 or len(top) == 2 or len(top) == 3

            detections_data = np.array(bottom[0].data)
            all_detections = self._read_detections(detections_data, MATCHED_RECORD_SIZE,
                                                   self._translate_matched_prediction,
                                                   self._valid_action_ids, self._min_conf)

            self._centers = np.array(bottom[1].data)

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

            max_intra_class_dist = 0.0
            losses = []
            valid_detections = []
            instance_counts = {}
            for det in all_detections:
                embedding = self._embeddings[det.anchor][det.item, :, det.y, det.x]

                if self._use_filtering:
                    scale = self._get_scale(self._step, self._init_scale, self._target_scale,
                                            self._num_steps, self._power, self._use_last)

                    scores = np.exp(scale * np.matmul(self._centers, embedding.reshape([-1, 1])))
                    distribution = scores / np.sum(scores)

                    regularization_value = -np.log(distribution[det.action]) +\
                                           self._entropy_weight * np.sum(distribution * np.log(distribution))

                    if regularization_value < 0.0:
                        continue

                center = self._centers[det.action]

                dist = 1.0 - np.sum(embedding * center)
                max_intra_class_dist = max(max_intra_class_dist, dist)

                losses.append(dist)
                valid_detections.append(det)

                if det.item not in instance_counts:
                    instance_counts[det.item] = {}
                local_instance_counts = instance_counts[det.item]
                if det.id not in local_instance_counts:
                    local_instance_counts[det.id] = 0
                local_instance_counts[det.id] += 1

            if self._instance_norm:
                instance_weights = [class_weights[det.action] / float(instance_counts[det.item][det.id])
                                    for det in valid_detections]
                num_instances = np.sum([len(counts) for counts in itervalues(instance_counts)])
            else:
                instance_weights = [class_weights[det.action] for det in valid_detections]
                num_instances = len(valid_detections)

            weighted_sum_losses = np.sum([instance_weights[i] * losses[i] for i, _ in enumerate(valid_detections)])

            top[0].data[...] = weighted_sum_losses / float(num_instances) if num_instances > 0 else 0.0
            if len(top) > 1:
                top[1].data[...] = max_intra_class_dist
                if len(top) == 3:
                    top[2].data[...] =\
                        float(len(valid_detections)) / float(len(all_detections)) if len(all_detections) > 0 else 0.0

            self._valid_detections = valid_detections
            self._weights = instance_weights
            self._num_instances = num_instances

            self._step += 1
        except Exception:
            LOG('CenterLossLayer forward pass exception: {}'.format(traceback.format_exc()))
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

            centers_diff_data = np.zeros(bottom[1].data.shape) if propagate_down[1] else None

            anchor_diff_data = {}
            for anchor_id in range(self._num_anchors):
                if propagate_down[anchor_id + 2]:
                    anchor_diff_data[anchor_id] = np.zeros(bottom[anchor_id + 2].data.shape)

            if len(self._valid_detections) > 0:
                factor = top[0].diff[0] / float(self._num_instances)

                for i, _ in enumerate(self._valid_detections):
                    det = self._valid_detections[i]
                    weight = -factor * self._weights[i]

                    if propagate_down[det.anchor + 2]:
                        anchor_diff_data[det.anchor][det.item, :, det.y, det.x]\
                            += weight * self._centers[det.action]

                    if centers_diff_data is not None:
                        centers_diff_data[det.action]\
                            += weight * self._embeddings[det.anchor][det.item, :, det.y, det.x]

            if centers_diff_data is not None:
                bottom[1].diff[...] = centers_diff_data

            for anchor_id in range(self._num_anchors):
                if propagate_down[anchor_id + 2]:
                    bottom[anchor_id + 2].diff[...] = anchor_diff_data[anchor_id]
        except Exception:
            LOG('CenterLossLayer backward pass exception: {}'.format(traceback.format_exc()))
            exit()

    def reshape(self, bottom, top):
        """Carry out blob reshaping.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        top[0].reshape(1)

        if len(top) > 1:
            top[1].reshape(1)
            if len(top) == 3:
                top[2].reshape(1)

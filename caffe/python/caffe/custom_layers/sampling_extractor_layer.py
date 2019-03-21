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


BBoxDesc = namedtuple('BBoxDesc', 'item, id, det_conf, anchor, action, x, y')

MATCHED_RECORD_SIZE = 11


class SamplingExtractorLayer(BaseLayer):
    """Extracts new samples (embeddings) by mixing random pair of border samples.
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

        detections = {i: [] for i, _ in enumerate(valid_action_ids)}
        for record in records:
            detection = converter(record)

            if detection.det_conf < min_conf or detection.item < 0:
                continue

            if detection.action in valid_action_ids:
                detections[detection.action].append(detection)

        return detections

    @staticmethod
    def _calc_scales_to_mix_embeddings(ratio, embd_x, embd_y):
        cos_gamma = np.sum(embd_x * embd_y, axis=1)
        sin_gamma = np.sqrt(1.0 - np.square(cos_gamma))

        alpha_angle = np.arccos((1. - ratio) / np.sqrt(ratio * ratio - 2.0 * ratio * cos_gamma + 1.0)) + \
                      np.arctan(ratio * sin_gamma / (ratio * cos_gamma - 1.0))
        gamma_angle = np.arccos(cos_gamma)
        betta_angle = gamma_angle - alpha_angle

        embd_x = np.sin(betta_angle) / sin_gamma
        embd_y = np.sin(alpha_angle) / sin_gamma

        return embd_x, embd_y

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
        self._num_steps = layer_params['num_steps'] if 'num_steps' in layer_params else 100
        assert self._num_steps > 0
        self._quantile = layer_params['quantile'] if 'quantile' in layer_params else 0.75
        assert 0.0 <= self._quantile <= 1.0
        self._threshold = layer_params['threshold'] if 'threshold' in layer_params else None

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
            assert len(top) == 2

            embedding_size = bottom[1].data.shape[1]

            detections_data = np.array(bottom[0].data)
            all_detections = self._read_detections(detections_data, MATCHED_RECORD_SIZE,
                                                   self._translate_matched_prediction,
                                                   self._valid_action_ids, self._min_conf)

            self._embeddings = []
            for i in range(self._num_anchors):
                self._embeddings.append(np.array(bottom[i + 1].data))

            valid_class_ids = list(all_detections)
            valid_class_ids.sort()

            self._samples = []
            sampled_vectors = []
            sampled_labels = []

            for class_id in valid_class_ids:
                detections = all_detections[class_id]
                if len(detections) < 2:
                    continue

                embeddings = np.array([self._embeddings[det.anchor][det.item, :, det.y, det.x] for det in detections])

                center = np.mean(embeddings, axis=0, keepdims=True)
                center = center / np.sqrt(np.sum(np.square(center), axis=1, keepdims=True))

                distances = (1.0 - np.matmul(embeddings, center.T)).reshape([-1])
                threshold = np.percentile(distances, 100.0 * self._quantile)\
                    if self._threshold is None else self._threshold
                valid_ids = np.arange(len(distances), dtype=np.int32)[distances > threshold]
                if len(valid_ids) < 2:
                    continue

                sample_ids1 = []
                sample_ids2 = []
                for _ in range(self._num_steps):
                    ids_pair = np.random.choice(valid_ids, 2, replace=False)
                    sample_ids1.append(ids_pair[0])
                    sample_ids2.append(ids_pair[1])
                dist_ratio = np.random.uniform(0.0, 1.0, [self._num_steps])
                alpha, betta = self._calc_scales_to_mix_embeddings(
                    dist_ratio, embeddings[sample_ids1], embeddings[sample_ids2])
                alpha = alpha.reshape([self._num_steps, 1])
                betta = betta.reshape([self._num_steps, 1])

                sample = alpha * embeddings[sample_ids1] + betta * embeddings[sample_ids2]
                sampled_vectors.append(sample)
                sampled_labels.append(np.full([self._num_steps], float(class_id), dtype=np.float32))

                self._samples += [(alpha[i], betta[i], detections[sample_ids1[i]], detections[sample_ids2[i]])
                                  for i, _ in enumerate(sample_ids1)]

            assert len(self._samples) == len(sampled_vectors) * self._num_steps

            if len(self._samples) > 0:
                top[0].reshape(len(self._samples), embedding_size)
                top[1].reshape(len(self._samples))

                top[0].data[...] = np.concatenate(tuple(sampled_vectors), axis=0)
                top[1].data[...] = np.concatenate(tuple(sampled_labels), axis=0)
            else:
                LOG('No samples generated!')

                top[0].reshape(1, embedding_size)
                top[0].data[...] = 0.0

                top[1].reshape(1)
                top[1].data[...] = -1.0
        except Exception:
            LOG('SamplingExtractorLayer forward pass exception: {}'.format(traceback.format_exc()))
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

            anchor_diff_data = {}
            for anchor_id in range(self._num_anchors):
                if propagate_down[anchor_id + 1]:
                    anchor_diff_data[anchor_id] = np.zeros(bottom[anchor_id + 1].data.shape)

            if len(self._samples) > 0:
                diff_data = np.array(top[0].diff)

                for out_sample_id, _ in enumerate(self._samples):
                    alpha, betta, det_i, det_j = self._samples[out_sample_id]

                    current_diff = diff_data[out_sample_id]

                    if propagate_down[det_i.anchor + 1]:
                        anchor_diff_data[det_i.anchor][det_i.item, :, det_i.y, det_i.x] \
                            += alpha * current_diff

                    if propagate_down[det_j.anchor + 1]:
                        anchor_diff_data[det_j.anchor][det_j.item, :, det_j.y, det_j.x] \
                            += betta * current_diff

            for anchor_id in range(self._num_anchors):
                if propagate_down[anchor_id + 1]:
                    bottom[anchor_id + 1].diff[...] = anchor_diff_data[anchor_id]
        except Exception:
            LOG('SamplingExtractorLayer backward pass exception: {}'.format(traceback.format_exc()))
            exit()

    def reshape(self, bottom, top):
        """Carry out blob reshaping.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        assert len(bottom) == (self._num_anchors + 1)
        assert len(top) == 2

        embedding_size = bottom[1].data.shape[1]

        top[0].reshape(1, embedding_size)
        top[1].reshape(1)

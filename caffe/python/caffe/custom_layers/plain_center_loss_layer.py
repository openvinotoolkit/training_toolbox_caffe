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

import numpy as np

from caffe._caffe import log as LOG
from caffe._caffe import Layer as BaseLayer


class PlainCenterLossLayer(BaseLayer):
    """One of Metric-learning losses which forces samples from the same class
       to be close to their center.
    """

    def _load_params(self, param_str):
        """Loads layer parameters.

        :param param_str: Input str of parameters
        """

        layer_params = eval(param_str)

        self._valid_class_ids = layer_params['valid_class_ids'] if 'valid_class_ids' in layer_params else None

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
            assert len(bottom) == 3
            assert len(top) == 1

            centers = np.array(bottom[0].data, dtype=np.float32)
            assert len(centers.shape) == 2

            embeddings = np.array(bottom[1].data, dtype=np.float32)
            assert len(embeddings.shape) == 2

            labels = np.array(bottom[2].data).astype(np.int32)
            assert len(labels.shape) == 1

            assert centers.shape[1] == embeddings.shape[1]
            assert embeddings.shape[0] == labels.shape[0]

            input_classes = np.unique(labels)

            sum_losses = 0.0
            num_instances = 0
            valid_class_pairs = []
            for class_id in input_classes:
                if class_id < 0 or self._valid_class_ids is not None and class_id not in self._valid_class_ids:
                    continue

                mask = labels == class_id
                center = centers[class_id].reshape([-1, 1])

                class_embeddings = embeddings[mask]

                distances = 1.0 - np.matmul(class_embeddings, center)
                sum_losses += np.sum(distances)

                num_instances += int(np.sum(mask))
                valid_class_pairs.append((class_id, mask))

            top[0].data[...] = sum_losses / float(num_instances) if num_instances > 0 else 0.0

            self._num_instances = num_instances
            self._valid_class_pairs = valid_class_pairs
        except Exception:
            LOG('PlainCenterLossLayer forward pass exception: {}'.format(traceback.format_exc()))
            exit()

    def backward(self, top, propagate_down, bottom):
        """Carry out backward pass.

        :param top: List of top blobs
        :param propagate_down: List of indicators to carry out back-propagation for
                               the specified bottom blob
        :param bottom: List of bottom blobs
        """

        try:
            if propagate_down[2]:
                raise Exception('Cannot propagate down through the labels')

            centers = np.array(bottom[0].data, dtype=np.float32)
            embeddings = np.array(bottom[1].data, dtype=np.float32)

            centers_diff_data = np.zeros(bottom[0].data.shape) if propagate_down[0] else None
            embeddings_diff_data = np.zeros(bottom[1].data.shape) if propagate_down[1] else None

            factor = top[0].diff[0] / float(self._num_instances)
            for class_id, mask in self._valid_class_pairs:
                if propagate_down[0]:
                    centers_diff_data[class_id] += -factor * np.sum(embeddings[mask], axis=0)

                if propagate_down[1]:
                    embeddings_diff_data[mask] += -factor * centers[class_id]

            if centers_diff_data is not None:
                bottom[0].diff[...] = centers_diff_data

            if embeddings_diff_data is not None:
                bottom[1].diff[...] = embeddings_diff_data
        except Exception:
            LOG('PlainCenterLossLayer backward pass exception: {}'.format(traceback.format_exc()))
            exit()

    def reshape(self, bottom, top):
        """Carry out blob reshaping.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        top[0].reshape(1)

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

import numpy as np

from caffe._caffe import log as LOG
from caffe._caffe import Layer as BaseLayer


class AdaptiveWeightingLossLayer(BaseLayer):
    """Layer for adaptive weighting between the input losses."""

    def _load_params(self, param_str, num_variables):
        """Loads layer parameters.

        :param param_str: Input str of parameters
        """

        layer_params = eval(param_str)

        self._scale = float(layer_params['scale']) if 'scale' in layer_params else 1.0
        self._init = layer_params['init'] if 'init' in layer_params else 0.0

        self._weights = layer_params['weights'] if 'weights' in layer_params else None
        if self._weights is None:
            self._weights = np.ones([num_variables], dtype=np.float32)
        else:
            assert len(self._weights) == num_variables
            assert np.all([w > 0.0 for w in self._weights])

    def _create_variables(self, num_params, init_value):
        """Initializes internal state"""

        self.blobs.add_blob(num_params)
        self.blobs[0].data[...] = init_value

    def setup(self, bottom, top):
        """Initializes layer.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        try:
            self._load_params(self.param_str, num_variables=len(bottom))

            num_variables = len(bottom)
            self._create_variables(num_variables, self._init)
        except Exception:
            LOG('AdaptiveWeightingLossLayer setup exception: {}'.format(traceback.format_exc()))
            exit()

    def forward(self, bottom, top):
        """Carry out forward pass.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        try:
            num_variables = len(bottom)
            assert num_variables > 0
            assert len(top) == 1 or len(top) == 1 + num_variables

            samples = []
            losses = []
            for i in range(num_variables):
                loss_value = np.array(bottom[i].data, dtype=np.float32).reshape([-1])
                assert len(loss_value) == 1

                loss_value = loss_value[0]

                if loss_value > 0.0:
                    param_value = self.blobs[0].data[i]
                    loss_factor = np.exp(-param_value)
                    new_loss_value = param_value + self._scale * loss_factor * loss_value

                    samples.append((i, self._scale * loss_factor, self._scale * loss_factor * loss_value))
                    losses.append(self._weights[i] * new_loss_value)

            top[0].data[...] = np.sum(losses) if len(losses) > 0 else 0.0

            if len(top) == 1 + num_variables:
                for i in range(num_variables):
                    top[i + 1].data[...] = np.copy(bottom[i].data)

            self._samples = samples
        except Exception:
            LOG('AdaptiveWeightingLossLayer forward pass exception: {}'.format(traceback.format_exc()))
            exit()

    def backward(self, top, propagate_down, bottom):
        """Carry out backward pass.

        :param top: List of top blobs
        :param propagate_down: List of indicators to carry out back-propagation for
                               the specified bottom blob
        :param bottom: List of bottom blobs
        """

        try:
            num_variables = len(bottom)
            for i in range(num_variables):
                bottom[i].diff[...] = 0.0

            top_diff_value = top[0].diff[0]
            for i, loss_scale, var_scale in self._samples:
                if propagate_down[i]:
                    bottom[i].diff[...] = self._weights[i] * loss_scale * top_diff_value

                    self.blobs[0].diff[i] += self._weights[i] * (1.0 - var_scale) * top_diff_value
        except Exception:
            LOG('AdaptiveWeightingLossLayer backward pass exception: {}'.format(traceback.format_exc()))
            exit()

    def reshape(self, bottom, top):
        """Carry out blob reshaping.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        top[0].reshape(1)

        num_variables = len(bottom)
        if len(top) == 1 + num_variables:
            for i in range(num_variables):
                top[i + 1].reshape(1)

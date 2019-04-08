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


class ScheduledScaleLayer(BaseLayer):
    """Layer to scale input blob by scheduled scalar value.
    """

    def _load_params(self, param_str):
        """Loads layer parameters.

        :param param_str: Input str of parameters
        """

        layer_params = eval(param_str)

        self._init_scale = layer_params['init_scale'] if 'init_scale' in layer_params else 16
        self._target_scale = layer_params['target_scale'] if 'target_scale' in layer_params else 1.5
        self._num_steps = layer_params['num_steps'] if 'num_steps' in layer_params else 140000
        self._power = layer_params['power'] if 'power' in layer_params else 2.0
        self._use_last = layer_params['use_last'] if 'use_last' in layer_params else False

    def _init_states(self):
        """Initializes internal state"""

        self._step = 0

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
            assert len(bottom) == 1
            assert len(top) == 1 or len(top) == 2

            data = np.array(bottom[0].data, dtype=np.float32)

            scale = self._get_scale(self._step, self._init_scale, self._target_scale,
                                    self._num_steps, self._power, self._use_last)
            self._last_scale = scale

            top[0].data[...] = scale * data

            if len(top) == 2:
                top[1].data[...] = scale

            self._step += 1
        except Exception:
            LOG('ScheduledScaleLayer forward pass exception: {}'.format(traceback.format_exc()))
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
                bottom[0].diff[...] = self._last_scale * np.copy(top[0].diff)
        except Exception:
            LOG('ScheduledScaleLayer backward pass exception: {}'.format(traceback.format_exc()))
            exit()

    def reshape(self, bottom, top):
        """Carry out blob reshaping.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        out_shape = bottom[0].data.shape
        top[0].reshape(*out_shape)

        if len(top) == 2:
            top[1].reshape(1)

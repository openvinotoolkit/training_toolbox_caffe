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

import numpy as np
import traceback
import cv2
from os.path import join, exists
from os import makedirs
from shutil import rmtree
from collections import namedtuple

from caffe._caffe import log as LOG
from caffe._caffe import Layer as BaseLayer


RECORD_SIZE = 10
ACTION_COLORS_MAP = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
ACTION_CV_COLORS_MAP = {0: 1, 1: 0, 2: 2}
ACTION_NAMES_MAP = {0: 'sit', 1: 'stand', 2: 'hand'}

BBoxDesc = namedtuple('BBoxDesc', 'item, anchor, xmin, ymin, xmax, ymax, x, y')


class ActionMapVisLayer(BaseLayer):
    @staticmethod
    def _translate_prediction(record):
        bbox = BBoxDesc(item=int(record[0]),
                        anchor=int(record[7]),
                        xmin=float(record[3]),
                        ymin=float(record[4]),
                        xmax=float(record[5]),
                        ymax=float(record[6]),
                        x=int(record[8]),
                        y=int(record[9]))
        return bbox

    @staticmethod
    def _read_detections(data, record_size, converter):
        assert data.size % record_size == 0, 'incorrect record_size'
        records = data.reshape([-1, record_size])

        detections = {}
        for record in records:
            detection = converter(record)
            detections[detection.item] = detections.get(detection.item, []) + [detection]

        return detections

    @staticmethod
    def _attention_map(embeddings, centers):
        height, width = embeddings.shape[1:]
        num_centers = centers.shape[0]
        assert num_centers == 3

        likelihood_maps = []
        max_likelihood_value = 1.0
        for center_id in xrange(num_centers):
            center = centers[center_id].reshape([-1, 1, 1])

            likelihood = np.sum(embeddings * center, axis=0)
            likelihood[likelihood < 0.0] = 0.0
            max_likelihood_value = np.maximum(max_likelihood_value, np.max(likelihood))

            likelihood_maps.append(likelihood)

        scale = 255. / float(max_likelihood_value)
        # scale = 255.

        colored_map = np.zeros([height, width, num_centers], dtype=np.uint8)
        for center_id in xrange(num_centers):
            float_likelihood_map = scale * likelihood_maps[center_id]
            int_map = float_likelihood_map.astype(np.uint8)
            colored_map[:, :, ACTION_CV_COLORS_MAP[center_id]] = int_map

        return colored_map

    @staticmethod
    def _draw_detections(src_image, detections):
        trg_image = np.copy(src_image)
        for detection in detections:
            xmin = int(round(detection.xmin * trg_image.shape[1]))
            ymin = int(round(detection.ymin * trg_image.shape[0]))
            xmax = int(round(detection.xmax * trg_image.shape[1]))
            ymax = int(round(detection.ymax * trg_image.shape[0]))

            cv2.rectangle(trg_image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)

            caption = '{}'.format(detection.anchor)
            cv2.putText(trg_image, caption, (xmin, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return trg_image

    def _load_params(self, param_str):
        pass
        layer_params = eval(param_str)

        assert 'num_anchors' in layer_params
        assert 'out_path' in layer_params

        self._num_anchors = layer_params['num_anchors']
        self._out_path = layer_params['out_path']

        if exists(self._out_path):
            rmtree(self._out_path)
        makedirs(self._out_path)

    def _init_states(self):
        self._iter = 0

    def setup(self, bottom, top):
        self._load_params(self.param_str)
        self._init_states()

    def forward(self, bottom, top):
        try:
            assert len(bottom) == self._num_anchors + 2 or len(bottom) == self._num_anchors + 3
            assert len(top) == 0

            image_data = np.array(bottom[0].data)
            centers_data = np.array(bottom[1].data)

            embeddings = []
            for i in xrange(self._num_anchors):
                embeddings.append(np.array(bottom[i + 2].data))

            if len(bottom) == self._num_anchors + 3:
                detections_data = np.array(bottom[self._num_anchors + 2].data)
                batch_detections = self._read_detections(detections_data, RECORD_SIZE,
                                                         self._translate_prediction)
            else:
                batch_detections = None

            for item_id in xrange(image_data.shape[0]):
                image = image_data[item_id].transpose([1, 2, 0]).astype(np.uint8)
                image_height, image_width = image.shape[:2]

                if batch_detections is not None:
                    detections = batch_detections[item_id]
                    annotated_image = self._draw_detections(image, detections)
                else:
                    annotated_image = image

                maps = []
                for anchor_id in xrange(self._num_anchors):
                    attention_map = self._attention_map(embeddings[anchor_id][item_id], centers_data)
                    maps.append(attention_map)

                assert self._num_anchors == 4
                out_attention_map = np.concatenate((np.concatenate((maps[0], maps[1]), axis=1),
                                                    np.concatenate((maps[2], maps[3]), axis=1)), axis=0)

                map_width = image_width
                map_height = int(float(out_attention_map.shape[0]) / float(out_attention_map.shape[1]) * map_width)
                out_attention_map = cv2.resize(out_attention_map, (map_width, map_height))

                out_image = np.concatenate((annotated_image, out_attention_map), axis=0)

                image_name = 'frame_{:04}_item_{:02}.png'.format(self._iter, item_id)
                cv2.imwrite(join(self._out_path, image_name), out_image)

            self._iter += 1
        except Exception:
            LOG('ActionMapVisLayer forward pass exception: {}'.format(traceback.format_exc()))
            exit()

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass


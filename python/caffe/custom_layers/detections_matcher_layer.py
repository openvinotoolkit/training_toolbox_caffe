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
from scipy.optimize import linear_sum_assignment

from caffe._caffe import log as LOG
from caffe._caffe import Layer as BaseLayer


InputDetection = namedtuple('InputDetection', 'item_id, track_id, real_action,'
                                              'det_conf, anchor_id,'
                                              'xmin, ymin, xmax, ymax,'
                                              'x_pos, y_pos')
MatchedDetection = namedtuple('MatchedDetection', 'item_id, track_id, real_action,'
                                                  'det_conf, anchor_id,'
                                                  'xmin, ymin, xmax, ymax,'
                                                  'pr_xmin, pr_ymin, pr_xmax, pr_ymax,'
                                                  'x_pos, y_pos')
BBox = namedtuple('BBox', 'xmin, ymin, xmax, ymax')

PREDICTION_RECORD_SIZE = 10
PREDICTION_BATCH_ID_POS = 0
GT_RECORD_SIZE = 8
GT_BATCH_ID_POS = 0
PRIOR_BOXES_RECORD_SIZE = 4


class DetMatcherLayer(BaseLayer):
    """Layer to match predicted bounding boxes with ground truth.
    """

    @staticmethod
    def _split_to_batch_gt(data, record_size, valid_action_ids):
        """Reads input blob with ground-truth annotation and splits it by batch ID.

        :param data: Input blob with annotation in ground-truth format
        :param record_size: Size of input record
        :return: Splitted by batch ID list of gt bboxes
        """

        assert data.size % record_size == 0, 'incorrect record_size'
        records = data.reshape([-1, record_size])

        if records.shape[0] == 0:
            return {}

        batch_size = int(np.max(records[:, 0])) + 1
        batch_data = {i: [] for i in range(batch_size)}

        for record in records:
            item_id = int(record[0])
            if item_id < 0:
                continue

            action_id = int(record[1])
            if action_id not in valid_action_ids:
                continue

            detection = InputDetection(item_id=item_id,
                                       track_id=int(record[2]),
                                       real_action=action_id,
                                       det_conf=1.0,
                                       anchor_id=-1,
                                       xmin=float(record[3]),
                                       ymin=float(record[4]),
                                       xmax=float(record[5]),
                                       ymax=float(record[6]),
                                       x_pos=-1,
                                       y_pos=-1)
            batch_data[detection.item_id].append(detection)

        return batch_data

    @staticmethod
    def _split_to_batch_prediction(data, record_size, priors, use_priors=False):
        """Reads input blob with net predictions and splits it by batch ID.

        :param data: Input blob with bbox descriptions
        :param record_size: Size of input record
        :param priors: Input blob with prior bboxes
        :param use_priors: Whether to use prior bboxes for matching with gt
        :return: Splitted by batch ID list of predicted bboxes
        """

        assert data.size % record_size == 0, 'incorrect record_size'
        records = data.reshape([-1, record_size])

        if records.shape[0] == 0:
            return {}

        batch_size = int(np.max(records[:, 0])) + 1
        batch_data = {i: [] for i in range(batch_size)}

        for record in records:
            item_id = int(record[0])
            if item_id < 0:
                continue

            anchor_id = int(record[7])
            x_pos = int(record[8])
            y_pos = int(record[9])

            if use_priors:
                out_bbox = priors[y_pos, x_pos, anchor_id]
            else:
                out_bbox = [float(record[3]), float(record[4]), float(record[5]), float(record[6])]

            detection = MatchedDetection(item_id=item_id,
                                         track_id=-1,
                                         real_action=-1,
                                         det_conf=float(record[2]),
                                         anchor_id=anchor_id,
                                         pr_xmin=float(record[3]), pr_ymin=float(record[4]),
                                         pr_xmax=float(record[5]), pr_ymax=float(record[6]),
                                         xmin=out_bbox[0], ymin=out_bbox[1],
                                         xmax=out_bbox[2], ymax=out_bbox[3],
                                         x_pos=x_pos, y_pos=y_pos)
            batch_data[detection.item_id].append(detection)

        return batch_data

    @staticmethod
    def _parse_prior_boxes(data, height, width, num_anchors, record_size):
        """Read blob with prior bbox descriptions.

        :param data: Input blob
        :param height: Height of target feature map
        :param width: Width of target feature map
        :param num_anchors: Number of anchor branches
        :param record_size: Size of input record
        :return: List of prior bboxes
        """

        data = data[0, 0]

        assert data.size % record_size == 0, 'incorrect record_size'
        assert data.size == height * width * num_anchors * record_size, 'incorrect record_size'

        return data.reshape([height, width, num_anchors, record_size])

    @staticmethod
    def _match_gt_to_predictions(gt_data, predicted_data, min_gt_iou, min_pr_iou, ssd_matching):
        """Carry out matching between set of ground-truth bboxes and
           predicted by network bboxes.

        :param gt_data: List of ground-truth bbxoes
        :param predicted_data: List of predicted bboxes
        :param min_gt_iou: Minimal IoU metric value to match gt onto prediction
        :param min_pr_iou: Minimal IoU metric value to match prediction with gt
        :param ssd_matching: Whether to use back matching: from prediction set to gt set
        :return: List of matched with gt detections
        """

        def _distance_matrix(set_a, set_b):
            """Measures the distance matrix between specified sets of bboxes.

            :param set_a: First set of bboxes
            :param set_b: Second set of bboxes
            :return: Distance matrix [len(set_a), len(set_b)]
            """

            def _extract_bbox_as_vector(bbox_set):
                """Repacks coordinates from list of values to solid vectors [n, 1].

                :param bbox_set: List of input bboxes
                :return: Solid vectors
                """

                return BBox(xmin=np.array([b.xmin for b in bbox_set], dtype=np.float32).reshape([-1, 1]),
                            ymin=np.array([b.ymin for b in bbox_set], dtype=np.float32).reshape([-1, 1]),
                            xmax=np.array([b.xmax for b in bbox_set], dtype=np.float32).reshape([-1, 1]),
                            ymax=np.array([b.ymax for b in bbox_set], dtype=np.float32).reshape([-1, 1]))

            def _extract_bbox_as_covector(bbox_set):
                """Repacks coordinates from list of values to solid co-vectors [1, n].

                :param bbox_set: List of input bboxes
                :return: Solid co-vectors
                """

                return BBox(xmin=np.array([b.xmin for b in bbox_set], dtype=np.float32).reshape([1, -1]),
                            ymin=np.array([b.ymin for b in bbox_set], dtype=np.float32).reshape([1, -1]),
                            xmax=np.array([b.xmax for b in bbox_set], dtype=np.float32).reshape([1, -1]),
                            ymax=np.array([b.ymax for b in bbox_set], dtype=np.float32).reshape([1, -1]))

            bboxes_a = _extract_bbox_as_vector(set_a)
            bboxes_b = _extract_bbox_as_covector(set_b)

            top_left_x = np.maximum(bboxes_a.xmin, bboxes_b.xmin)
            top_left_y = np.maximum(bboxes_a.ymin, bboxes_b.ymin)

            intersect_width = np.maximum(0.0, np.minimum(bboxes_a.xmax, bboxes_b.xmax) - top_left_x)
            intersect_height = np.maximum(0.0, np.minimum(bboxes_a.ymax, bboxes_b.ymax) - top_left_y)
            intersection_area = intersect_width * intersect_height

            area1 = (bboxes_a.xmax - bboxes_a.xmin) * (bboxes_a.ymax - bboxes_a.ymin)
            area2 = (bboxes_b.xmax - bboxes_b.xmin) * (bboxes_b.ymax - bboxes_b.ymin)

            area1[np.less(area1, 0.0)] = 0.0
            area2[np.less(area2, 0.0)] = 0.0

            union_area = area1 + area2 - intersection_area

            overlaps = intersection_area / union_area
            overlaps[np.less_equal(union_area, 0.0)] = 0.0

            return 1.0 - overlaps

        matched_detections = {}
        for item_id in gt_data:
            if item_id >= len(predicted_data):
                continue

            gt_bboxes = gt_data[item_id]
            predicted_bboxes = predicted_data[item_id]

            if len(gt_bboxes) == 0 or len(predicted_bboxes) == 0:
                continue

            matches = []
            matched_detections_mask = np.zeros([len(predicted_bboxes)], dtype=np.bool)

            distance_matrix = _distance_matrix(gt_bboxes, predicted_bboxes)

            # First: match gt to predictions 1:1 with IoU > min_gt_iou
            gt_inds, pred_inds = linear_sum_assignment(distance_matrix)

            for i, _ in enumerate(gt_inds):
                gt_id = gt_inds[i]
                predicted_id = pred_inds[i]

                matched_iou = 1. - distance_matrix[gt_id, predicted_id]
                if matched_iou <= min_gt_iou:
                    continue

                matched_detections_mask[predicted_id] = True

                gt_bbox = gt_bboxes[gt_id]
                predicted_bbox = predicted_bboxes[predicted_id]

                matches.append(InputDetection(item_id=item_id,
                                              track_id=gt_bbox.track_id,
                                              real_action=gt_bbox.real_action,
                                              det_conf=predicted_bbox.det_conf,
                                              anchor_id=predicted_bbox.anchor_id,
                                              xmin=predicted_bbox.pr_xmin,
                                              ymin=predicted_bbox.pr_ymin,
                                              xmax=predicted_bbox.pr_xmax,
                                              ymax=predicted_bbox.pr_ymax,
                                              x_pos=predicted_bbox.x_pos,
                                              y_pos=predicted_bbox.y_pos))

            # Second: match prediction to gt with IoU > min_pr_iou
            if ssd_matching:
                best_matched_pr_to_gt_id = np.argmin(distance_matrix, axis=0)
                best_matched_pr_to_gt_val = np.min(distance_matrix, axis=0)

                valid_mask =\
                    (best_matched_pr_to_gt_val < 1.0 - min_pr_iou) * np.logical_not(matched_detections_mask)
                valid_prediction_ids = np.arange(len(predicted_bboxes))[valid_mask]
                valid_matched_gt_ids = best_matched_pr_to_gt_id[valid_mask]

                valid_matched_gt_bboxes = [gt_bboxes[i] for i in valid_matched_gt_ids]
                valid_predicted_bboxes = [predicted_bboxes[i] for i in valid_prediction_ids]

                for i, _ in enumerate(valid_matched_gt_bboxes):
                    gt_bbox = valid_matched_gt_bboxes[i]
                    predicted_bbox = valid_predicted_bboxes[i]
                    matches.append(InputDetection(item_id=item_id,
                                                  track_id=gt_bbox.track_id,
                                                  real_action=gt_bbox.real_action,
                                                  det_conf=predicted_bbox.det_conf,
                                                  anchor_id=predicted_bbox.anchor_id,
                                                  xmin=predicted_bbox.pr_xmin,
                                                  ymin=predicted_bbox.pr_ymin,
                                                  xmax=predicted_bbox.pr_xmax,
                                                  ymax=predicted_bbox.pr_ymax,
                                                  x_pos=predicted_bbox.x_pos,
                                                  y_pos=predicted_bbox.y_pos))

            matched_detections[item_id] = matches

        return matched_detections

    @staticmethod
    def _convert_predictions_to_blob(predictions, record_size):
        """Carry out back conversation of matched bboxes to solid blob of data.

        :param predictions: List of matched bboxes
        :param record_size: Size of output record
        :return: Blob with converted data
        """

        records = []
        for item_id in predictions:
            matched_detections = predictions[item_id]

            for det in matched_detections:
                record = [item_id, det.det_conf,
                          det.xmin, det.ymin, det.xmax, det.ymax,
                          det.anchor_id, det.track_id, det.real_action,
                          det.x_pos, det.y_pos]
                assert len(record) == record_size

                records.append(record)

        out_blob = np.array(records, dtype=np.float32)
        out_blob = out_blob.reshape([1, 1, len(records), record_size])

        return out_blob

    def _load_params(self, param_str):
        """Loads layer parameters.

        :param param_str: Input str of parameters
        """

        layer_params = eval(param_str)

        assert 'height' in layer_params
        assert 'width' in layer_params
        assert 'num_anchors' in layer_params
        assert 'valid_action_ids' in layer_params

        self._height = layer_params['height']
        self._width = layer_params['width']
        self._num_anchors = layer_params['num_anchors']
        self._valid_action_ids = layer_params['valid_action_ids']
        assert len(self._valid_action_ids) > 0

        self._min_gt_iou = layer_params['min_gt_iou'] if 'min_gt_iou' in layer_params else 0.01
        self._min_pr_iou = layer_params['min_pr_iou'] if 'min_pr_iou' in layer_params else 0.5
        self._ssd_matching = layer_params['ssd_matching'] if 'ssd_matching' in layer_params else False

        self._match_priors = layer_params['match_priors'] if 'match_priors' in layer_params else False

        self._record_size = 11

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
            assert len(top) == 1 or len(top) == 2

            pred_detections = np.array(bottom[0].data)
            gt_detections = np.array(bottom[1].data)
            prior_boxes_data = np.array(bottom[2].data)

            batch_gt = self._split_to_batch_gt(gt_detections, GT_RECORD_SIZE, self._valid_action_ids)

            prior_boxes =\
                self._parse_prior_boxes(prior_boxes_data,
                                        self._height, self._width, self._num_anchors,
                                        PRIOR_BOXES_RECORD_SIZE)

            batch_predictions =\
                self._split_to_batch_prediction(pred_detections, PREDICTION_RECORD_SIZE, prior_boxes,
                                                use_priors=self._match_priors)

            matched_predictions = self._match_gt_to_predictions(batch_gt, batch_predictions,
                                                                min_gt_iou=self._min_gt_iou,
                                                                min_pr_iou=self._min_pr_iou,
                                                                ssd_matching=self._ssd_matching)

            matches_blob = self._convert_predictions_to_blob(matched_predictions, self._record_size)
            out_shape = matches_blob.shape

            if out_shape[2] == 0:
                LOG('!!!No matched detections!!!')
                top[0].reshape(out_shape[0], out_shape[1], 1, out_shape[3])
                top[0].data[...] = -1.0
                if len(top) == 2:
                    top[1].data[...] = 0.0
            else:
                top[0].reshape(out_shape[0], out_shape[1], out_shape[2], out_shape[3])
                top[0].data[...] = matches_blob

                if len(top) == 2:
                    num_predictions = np.sum([len(l) for l in itervalues(batch_predictions)])
                    num_matches = np.sum([len(l) for l in itervalues(matched_predictions)])
                    top[1].data[...] = float(num_matches) / float(num_predictions) if num_predictions > 0 else 0.0
        except Exception:
            LOG('MatcherLayer exception: {}'.format(traceback.format_exc()))
            exit()

    def backward(self, top, propagate_down, bottom):
        """Carry out backward pass.

        :param top: List of top blobs
        :param propagate_down: List of indicators to carry out back-propagation for
                               the specified bottom blob
        :param bottom: List of bottom blobs
        """

        pass

    def reshape(self, bottom, top):
        """Carry out blob reshaping.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        top[0].reshape(1, 1, 1, self._record_size)

        if len(top) == 2:
            top[1].reshape(1)

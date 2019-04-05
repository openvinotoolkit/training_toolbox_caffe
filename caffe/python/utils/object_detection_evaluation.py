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

from collections import defaultdict

import glog as log
import numpy as np
from six import itervalues

from .python_iou import iou_python as box_iou
from .python_iou import ioa as box_ioa
from .detection import voc_ap

class ImageDetectionEvaluation(object):
    def __init__(self, num_ground_truth_classes, iou_threshold,
                 prefer_regular_ground_truth=True,
                 allow_multiple_matches_per_ignored=True,
                 allow_multiple_matches_per_regular=False):
        self.num_ground_truth_classes = num_ground_truth_classes
        self.iou_threshold = iou_threshold
        self.prefer_regular_ground_truth = prefer_regular_ground_truth
        self.allow_multiple_matches_per_ignored = allow_multiple_matches_per_ignored
        self.allow_multiple_matches_per_regular = allow_multiple_matches_per_regular
        self.detected_boxes = None
        self.detected_scores = None
        self.detected_labels = None
        self.ground_truth_boxes = None
        self.ground_truth_labels = None
        self.ground_truth_ignorance_mask = None
        self.detections_matching = None
        self._ground_truth_matching = None
        self.is_ground_truth_of_interest = None
        self.__matching_required = True

    def set_detections(self, detected_boxes, detected_scores, detected_labels):
        if len(detected_boxes) != len(detected_scores) or len(detected_boxes) != len(detected_labels):
            raise ValueError('detected_boxes, detected_scores and '
                             'detected_class_labels should all have same lengths. Got'
                             '[{}, {}, {}]'.format(len(detected_boxes),
                                                   len(detected_scores),
                                                   len(detected_labels))
                             )
        assert len(detected_boxes.shape) == 2 and detected_boxes.shape[1] == 4
        assert len(detected_labels.shape) == 1
        assert len(detected_scores.shape) == 1
        self.detected_boxes = detected_boxes.copy()
        self.detected_labels = detected_labels.copy()
        self.detected_scores = detected_scores.copy()
        self.__matching_required = True

    def set_ground_truth(self, boxes, labels, is_ignored):
        if len(boxes) != len(labels) or len(labels) != len(is_ignored):
            raise ValueError('boxes, labels and ignorance mask should all have same lengths. Got'
                             '[{}, {}, {}]'.format(len(boxes), len(labels), len(is_ignored)))
        assert len(boxes.shape) == 2 and boxes.shape[1] == 4
        assert len(labels.shape) == 1
        self.ground_truth_boxes = boxes.copy()
        self.ground_truth_labels = labels.copy()
        self.ground_truth_ignorance_mask = is_ignored.copy()
        self.__matching_required = True

    def set_filter(self, is_ground_truth_of_interest):
        if is_ground_truth_of_interest is None or self.ground_truth_boxes is None:
            log.info(str(is_ground_truth_of_interest))
            log.info(str(self.ground_truth_boxes))
        assert is_ground_truth_of_interest is None or len(is_ground_truth_of_interest) == len(self.ground_truth_boxes)
        self.is_ground_truth_of_interest = is_ground_truth_of_interest

    def add_filter(self, is_ground_truth_of_interest):
        if not (is_ground_truth_of_interest is None or len(is_ground_truth_of_interest) == len(self.ground_truth_boxes)):
            log.info(str(is_ground_truth_of_interest))
            log.info(str(self.ground_truth_boxes))
        assert is_ground_truth_of_interest is None or len(is_ground_truth_of_interest) == len(self.ground_truth_boxes)
        self.is_ground_truth_of_interest = np.logical_and(self.is_ground_truth_of_interest, is_ground_truth_of_interest)

    def evaluate(self):
        if self.__matching_required:
            self.match_all()

        # gt_mask == mask of regular GT boxes of interest.
        gt_mask = ~self.ground_truth_ignorance_mask
        if self.is_ground_truth_of_interest is not None:
            gt_mask = np.logical_and(gt_mask, self.is_ground_truth_of_interest)

        # det_mask == mask of false positives or detections matched to regular GT boxes of interest.
        det_mask = self.detections_matching < 0
        if len(gt_mask) > 0:
            det_mask = np.logical_or(det_mask, gt_mask[self.detections_matching])

        # Mask of true positives / false negatives.
        is_true_positive = np.zeros(len(self.detected_boxes), dtype=bool)
        is_true_positive[self.detections_matching >= 0] = True

        # Leave only those detections that pass through det_mask.
        is_true_positive = is_true_positive[det_mask]
        scores = self.detected_scores[det_mask]
        labels = self.detected_labels[det_mask]

        # Count number of valid GT boxes per class.
        gt_boxes_num = [np.count_nonzero(np.logical_and(self.ground_truth_labels == class_idx,
                                                        gt_mask))
                        for class_idx in range(self.num_ground_truth_classes)]

        return is_true_positive, scores, labels, gt_boxes_num

    def positive_negative(self):
        if self.__matching_required:
            self.match_all()

        is_true_positive = np.zeros(len(self.detected_boxes), dtype=np.float32)

        matched_gts = np.unique(self.detections_matching)
        matched_gts_num = np.count_nonzero(matched_gts != -1)

        mask = self.detections_matching >= 0
        is_true_positive[mask] = -1

        idx = self.detections_matching[mask]
        non_ignorance_mask = ~self.ground_truth_ignorance_mask[idx]
        m = mask.copy()
        m[mask] = m[mask] & non_ignorance_mask
        is_true_positive[m] = 1

        corresponding_gt = np.ones((len(self.detections_matching), 4), dtype=np.float32)
        corresponding_gt[:, 2:] = 2
        corresponding_gt[mask] = self.ground_truth_boxes[idx]

        return is_true_positive, corresponding_gt, matched_gts_num, idx

    def match_all(self):
        self.detections_matching = np.empty(len(self.detected_boxes), np.int32)
        self.detections_matching.fill(-1)
        self._ground_truth_matching = None

        for label in range(self.num_ground_truth_classes):
            det_subset_mask = self.detected_labels == label
            det_subset_indices = np.where(det_subset_mask)[0]
            det_boxes = self.detected_boxes[det_subset_mask, :]
            det_scores = self.detected_scores[det_subset_mask]

            gt_subset_mask = self.ground_truth_labels == label
            gt_subset_indices = np.where(gt_subset_mask)[0]
            gt_boxes = self.ground_truth_boxes[gt_subset_mask, :]
            gt_ignorance_mask = self.ground_truth_ignorance_mask[gt_subset_mask]

            det_matching = self.match(det_boxes, det_scores, gt_boxes, gt_ignorance_mask)

            mask = det_matching >= 0
            self.detections_matching[det_subset_indices[mask]] = gt_subset_indices[det_matching[mask]]
        self.__matching_required = False

    @property
    def ground_truth_matching(self):
        assert self.detections_matching is not None, 'Do matching first.'
        if self._ground_truth_matching is None:
            self._ground_truth_matching = [[] for _ in enumerate(self.ground_truth_boxes)]
            for gt_idx, _ in enumerate(self.ground_truth_boxes):
                self._ground_truth_matching[gt_idx] = np.where(self.detections_matching == gt_idx)[0]
        return self._ground_truth_matching

    def match(self, det_boxes, det_scores, gt_boxes, gt_ignorance_mask):
        def add_fake_column(mat):
            fake_column = np.empty((mat.shape[0], 1), dtype=np.float32)
            result = np.concatenate((mat, fake_column), axis=1)
            result[:, -1].fill(-np.inf)
            return result

        # Order detections in score descending order.
        det_order = np.argsort(-det_scores)
        det_matching = np.empty(len(det_boxes), np.int32)
        det_matching.fill(-1)

        if det_boxes.size == 0 or gt_boxes.size == 0:
            return det_matching

        if self.prefer_regular_ground_truth:
            # Find overlaps with regular boxes and choose closest detections.
            gt_boxes_regular = gt_boxes[~gt_ignorance_mask, ...]
            iou_regular = box_iou(det_boxes, gt_boxes_regular)
            # Adding fake column to IoU matrix to properly handle the case when no regular GT boxes provided.
            iou_regular = add_fake_column(iou_regular)
            max_overlap_gt_ids = np.argmax(iou_regular, axis=1)

            # Find overlaps with ignored boxes and choose closest detections.
            gt_boxes_ignored = gt_boxes[gt_ignorance_mask, ...]
            if self.allow_multiple_matches_per_ignored:
                iou_ignored = box_ioa(gt_boxes_ignored, det_boxes).transpose()
            else:
                iou_ignored = box_iou(det_boxes, gt_boxes_ignored)
            iou_ignored = add_fake_column(iou_ignored)
            max_overlap_ignored_gt_ids = np.argmax(iou_ignored, axis=1)

            # For each detection (in sorted order) get the index of the closest GT box.
            gt_regular_id = max_overlap_gt_ids[det_order]
            # Get detection hits mask.
            regular_hits_mask = iou_regular[det_order, gt_regular_id] >= self.iou_threshold
            # Indices of GT boxes that are covered by detections.
            regular_gt_indices = np.where(~gt_ignorance_mask)[0]
            gts = regular_gt_indices[gt_regular_id[regular_hits_mask]]
            if self.allow_multiple_matches_per_regular:
                det_matching[det_order[regular_hits_mask]] = gts
            else:
                gts, gts_mask = np.unique(gts, return_index=True)
                det_matching[det_order[regular_hits_mask][gts_mask]] = gts

            gt_ignored_id = max_overlap_ignored_gt_ids[det_order]
            ignored_hits_mask = iou_ignored[det_order, gt_ignored_id] >= self.iou_threshold
            ignored_hits_mask = ignored_hits_mask & ~regular_hits_mask
            ignored_gt_indices = np.where(gt_ignorance_mask)[0]
            gts = ignored_gt_indices[gt_ignored_id[ignored_hits_mask]]
            if self.allow_multiple_matches_per_ignored:
                det_matching[det_order[ignored_hits_mask]] = gts
            else:
                gts, gts_mask = np.unique(gts, return_index=True)
                det_matching[det_order[ignored_hits_mask][gts_mask]] = gts
        else:
            # Find overlaps with GT boxes and choose closest detections.
            iou = box_iou(det_boxes, gt_boxes)
            # Adding fake column to IoU matrix to properly handle the case when no GT boxes provided.
            iou = add_fake_column(iou)
            max_overlap_gt_ids = np.argmax(iou, axis=1)

            # For each detection (in sorted order) get the index of the closest GT box.
            gt_id = max_overlap_gt_ids[det_order]
            # Get detection hits mask.
            hits_mask = iou[det_order, gt_id] >= self.iou_threshold
            # Indices of GT boxes that are covered by detections.
            gts = gt_id[hits_mask]
            if self.allow_multiple_matches_per_regular:
                det_matching[det_order[hits_mask]] = gts
            else:
                gts, gts_mask = np.unique(gts, return_index=True)
                det_matching[det_order[hits_mask][gts_mask]] = gts

        return det_matching


class DetectionEvaluation(object):
    def __init__(self, num_ground_truth_classes, iou_threshold,
                 prefer_regular_ground_truth=True,
                 allow_multiple_matches_per_ignored=True):
        self.dataset = defaultdict(lambda: ImageDetectionEvaluation(num_ground_truth_classes, iou_threshold,
                                                                    prefer_regular_ground_truth,
                                                                    allow_multiple_matches_per_ignored))
        self.num_classes = num_ground_truth_classes

    def set_detections(self, image_key, detected_boxes, detected_scores, detected_labels):
        self.dataset[image_key].set_detections(detected_boxes, detected_scores, detected_labels)

    def set_ground_truth(self, image_key, boxes, labels, is_ignored):
        self.dataset[image_key].set_ground_truth(boxes, labels, is_ignored)

    def set_ground_truth_of_interest(self, image_key, is_ground_truth_of_interest):
        self.dataset[image_key].set_filter(is_ground_truth_of_interest)

    def evaluate(self):
        is_true_positive = [np.empty(0, dtype=bool) for _ in range(self.num_classes)]
        detection_scores = [np.empty(0, dtype=np.float32) for _ in range(self.num_classes)]
        gt_boxes_per_class = [0 for _ in range(self.num_classes)]
        for image_annotation in itervalues(self.dataset):
            is_tp, scores, labels, gt_boxes_num = image_annotation.evaluate()
            for l in range(self.num_classes):
                is_true_positive[l] = np.concatenate((is_true_positive[l], is_tp[labels == l]))
                detection_scores[l] = np.concatenate((detection_scores[l], scores[labels == l]))
                gt_boxes_per_class[l] += gt_boxes_num[l]

        precisions_per_class = [[] for _ in range(self.num_classes)]
        recalls_per_class = [[] for _ in range(self.num_classes)]
        average_precision_per_class = [0.0 for _ in range(self.num_classes)]
        for class_index in range(self.num_classes):
            if len(is_true_positive[class_index]) == 0:
                continue

            def compute_tp_fp(labels, scores):
              sorted_ind = np.argsort(-scores)
              tp_labels = labels[sorted_ind]
              fp_labels = (tp_labels <= 0).astype(float)
              true_positives = np.cumsum(tp_labels)
              false_positives = np.cumsum(fp_labels)
              return true_positives, false_positives

            tp, fp = compute_tp_fp(is_true_positive[class_index], detection_scores[class_index])
            precision = tp.astype(float) / (tp + fp)
            recall = tp.astype(float) / gt_boxes_per_class[class_index]

            precisions_per_class.append(precision)
            recalls_per_class.append(recall)
            if precision is None and recall is None:  # No GT labels -> Recall = 1
                log.warning('No GT labels for class with idx = {0}'.format(class_index))
                num_detections = detection_scores[class_index].size
                if num_detections == 0:
                    average_precision = 1.
                else:
                    average_precision = 0.
            else:
                average_precision = voc_ap(recall, precision)
            average_precision_per_class[class_index] = average_precision

        mean_ap = np.nanmean(average_precision_per_class)
        return (average_precision_per_class, mean_ap,
                precisions_per_class, recalls_per_class)

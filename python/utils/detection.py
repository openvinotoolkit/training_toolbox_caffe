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
from collections import namedtuple
from bisect import bisect
from tqdm import tqdm


def voc_ap(recall, precision, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.
    else:
        # Correct AP calculation.
        # First append sentinel values at the end.
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # Compute the precision envelope.
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # To calculate area under PR curve, look for points
        # where X axis (recall) changes value.
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # And sum (\Delta recall) * prec.
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def miss_rate(miss_rates, fppis, fppi_level=0.1):
    position = bisect(fppis, fppi_level)
    p1 = position - 1
    p2 = position if position < len(miss_rates) else p1
    return 0.5 * (miss_rates[p1] + miss_rates[p2])


def evaluate_detections(ground_truth, predictions, class_name, overlap_threshold=0.5,
                        allow_multiple_matches_per_ignored=True,
                        verbose=True):
    Detection = namedtuple('Detection', ['image', 'bbox', 'score', 'gt_match'])
    GT = namedtuple('GroundTruth', ['bbox', 'is_matched', 'is_ignored'])
    detections = [Detection(image=img_pred.image_path,
                            bbox=np.array(obj_pred["bbox"]),
                            score=obj_pred.get("score", 0.0),
                            gt_match=-1)
                  for img_pred in predictions
                  for obj_pred in img_pred
                  if obj_pred["type"] == class_name]

    scores = np.array([detection.score for detection in detections])
    sorted_ind = np.argsort(-scores)
    detections = [detections[i] for i in sorted_ind]

    gts = {img_gt.image_path: GT(
        bbox=np.vstack([np.array(obj_gt["bbox"]) for obj_gt in img_gt]) if img_gt else np.empty((0, 4)),
        is_matched=np.zeros(len(img_gt), dtype=bool),
        is_ignored=np.array([obj_gt.get("is_ignored", False) for obj_gt in img_gt], dtype=bool))
           for img_gt in ground_truth}

    nd = len(detections)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for i, detection in tqdm(enumerate(detections), desc="Processing detections", disable=not verbose):
        image_path = detection.image
        bboxes_gt = gts[image_path].bbox
        bbox = detection.bbox
        max_overlap = -np.inf

        if bboxes_gt is not None and len(bboxes_gt) > 0:
            intersection_xmin = np.maximum(bboxes_gt[:, 0], bbox[0])
            intersection_ymin = np.maximum(bboxes_gt[:, 1], bbox[1])
            intersection_xmax = np.minimum(bboxes_gt[:, 0] + bboxes_gt[:, 2], bbox[0] + bbox[2])
            intersection_ymax = np.minimum(bboxes_gt[:, 1] + bboxes_gt[:, 3], bbox[1] + bbox[3])
            intersection_width = np.maximum(intersection_xmax - intersection_xmin, 0.)
            intersection_height = np.maximum(intersection_ymax - intersection_ymin, 0.)
            intersection = intersection_width * intersection_height

            det_area = bbox[2] * bbox[3]
            gt_area = bboxes_gt[:, 2] * bboxes_gt[:, 3]
            union = (det_area + gt_area - intersection)
            ignored_mask = gts[image_path].is_ignored
            if allow_multiple_matches_per_ignored:
                if np.any(ignored_mask):
                    union[ignored_mask] = det_area

            overlaps = intersection / union
            # Match not ignored ground truths first.
            if np.any(~ignored_mask):
                overlaps_filtered = np.copy(overlaps)
                overlaps_filtered[ignored_mask] = 0.0
                max_overlap = np.max(overlaps_filtered)
                argmax_overlap = np.argmax(overlaps_filtered)
            # If match with non-ignored ground truth is not good enough,
            # try to match with ignored ones.
            if max_overlap < overlap_threshold and np.any(ignored_mask):
                overlaps_filtered = np.copy(overlaps)
                overlaps_filtered[~ignored_mask] = 0.0
                max_overlap = np.max(overlaps_filtered)
                argmax_overlap = np.argmax(overlaps_filtered)
            detections[i] = detection._replace(gt_match=argmax_overlap)

        if max_overlap >= overlap_threshold:
            if not gts[image_path].is_ignored[argmax_overlap]:
                if not gts[image_path].is_matched[argmax_overlap]:
                    tp[i] = 1.
                    gts[image_path].is_matched[argmax_overlap] = True
                else:
                    fp[i] = 1.
            elif not allow_multiple_matches_per_ignored:
                gts[image_path].is_matched[argmax_overlap] = True
        else:
            fp[i] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    debug_visualization = False
    if debug_visualization:
        for im, bboxes_gt in gts.iteritems():
            import cv2
            print(im)
            image = cv2.imread(im)
            image_gt = np.copy(image)
            for b in bboxes_gt.bbox:
                cv2.rectangle(image_gt, tuple(b[:2]), tuple(b[2:] + b[:2]), color=(255, 255, 0), thickness=2)
            cv2.imshow("gt", image_gt)
            for detection in detections:
                if detection.image != im:
                    continue
                b = detection.bbox
                cv2.rectangle(image, tuple(b[:2]), tuple(b[2:] + b[:2]), color=(0, 255, 0), thickness=2)
                if detection.gt_match is not None:
                    b = bboxes_gt.bbox[detection.gt_match]
                    cv2.rectangle(image, tuple(b[:2]), tuple(b[2:] + b[:2]), color=(0, 0, 255), thickness=1)
                cv2.imshow("image", image)
                cv2.waitKey(0)

    # Handle equal-score detections.
    # Get index of the last occurrence of a score.
    ind = len(scores) - np.unique(scores[sorted_ind[::-1]], return_index=True)[1] - 1
    ind = ind[::-1]
    # Though away redundant points.
    fp = fp[ind]
    tp = tp[ind]

    total_positives_num = np.sum([np.count_nonzero(~gt.is_ignored) for gt in gts.values()])
    recall = tp / float(total_positives_num)
    # Avoid divide by zero in case the first detection matches an ignored ground truth.
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    miss_rate = 1.0 - recall
    fppi = fp / float(len(gts))

    return recall, precision, miss_rate, fppi

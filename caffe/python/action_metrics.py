#!/usr/bin/env python3

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

from __future__ import print_function

#pylint: disable=ungrouped-imports
import os
import json

from argparse import ArgumentParser
from bisect import bisect
from builtins import range
from collections import namedtuple
from os.path import exists, basename

import cv2
import numpy as np
from lxml import etree
from six import iteritems, itervalues
from tqdm import tqdm

os.environ['GLOG_minloglevel'] = '2'
#pylint: disable=wrong-import-position
import caffe


BBoxDesc = namedtuple('BBoxDesc', 'label, det_conf, action_conf, xmin, ymin, xmax, ymax')


def extract_video_properties(vidcap):
    """Extracts video height, width, fps and number of frames.

    :param vidcap: Input video source
    :return: Tuple with required video parameters
    """

    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    return height, width, fps, length


def parse_tasks(file_path):
    """Process input file to extract list of tasks. Each tasks is
       represented by line: "path_to_annotation path_to_video_file"

    :param file_path: Input file with tasks
    :return: Extracted list of tasks
    """

    print('Found tasks:')

    tasks = []
    data_dir = os.path.dirname(file_path)
    with open(file_path, 'r') as input_stream:
        for line in input_stream:
            if line.endswith('\n'):
                line = line[:-len('\n')]

            if len(line) == 0:
                continue

            annotation_path, video_path = line.split(' ')
            annotation_path = os.path.join(data_dir, annotation_path)
            video_path = os.path.join(data_dir, video_path)
            if not exists(annotation_path) or not exists(video_path):
                continue

            tasks.append((annotation_path, video_path))
            print('   #{}: {} {}'.format(len(tasks), annotation_path, video_path))

    return tasks


def load_annotation(annotation_path, video_size, action_names_map):
    """Loads annotation from the specified file.

    :param annotation_path: Path to file with annotation
    :param video_size: Input video height and width
    :return: Loaded annotation
    """

    tree = etree.parse(annotation_path)
    root = tree.getroot()

    detections_by_frame_id = {}
    roi = [float(video_size[1]), float(video_size[0]), 0.0, 0.0]

    for track in tqdm(root, desc='Extracting annotation'):
        if 'label' not in track.attrib or track.attrib['label'] != 'person':
            continue

        for bbox in track:
            if len(bbox) < 1:
                continue

            frame_id = int(bbox.attrib['frame'])

            action_name = None
            for bbox_attr_id, _ in enumerate(bbox):
                attribute_name = bbox[bbox_attr_id].attrib['name']
                if attribute_name != 'action':
                    continue

                action_name = bbox[bbox_attr_id].text

            if action_name is not None and action_name in action_names_map:
                label = action_names_map[action_name]

                xmin = float(bbox.attrib['xtl'])
                ymin = float(bbox.attrib['ytl'])
                xmax = float(bbox.attrib['xbr'])
                ymax = float(bbox.attrib['ybr'])

                roi = [min(xmin, roi[0]),
                       min(ymin, roi[1]),
                       max(xmax, roi[2]),
                       max(ymax, roi[3])]

                bbox_desc = BBoxDesc(label=label,
                                     det_conf=1.0,
                                     action_conf=1.0,
                                     xmin=xmin / float(video_size[1]),
                                     ymin=ymin / float(video_size[0]),
                                     xmax=xmax / float(video_size[1]),
                                     ymax=ymax / float(video_size[0]))

                detections_by_frame_id[frame_id] = detections_by_frame_id.get(frame_id, []) + [bbox_desc]

    int_roi = [max(0, int(roi[0])),
               max(0, int(roi[1])),
               min(int(video_size[1]), int(roi[2])),
               min(int(video_size[0]), int(roi[3]))]

    print('Loaded {} annotated frames.'.format(len(detections_by_frame_id)))

    return detections_by_frame_id, int_roi


def iou(box_a, box_b):
    """ Calculates Intersection over Union (IoU) metric.

    :param box_a: First bbox
    :param box_b: Second bbox
    :return: Scalar value of metric
    """

    top_left_x = max(box_a.xmin, box_b.xmin)
    top_left_y = max(box_a.ymin, box_b.ymin)
    intersect_width = max(0.0, min(box_a.xmax, box_b.xmax) - top_left_x)
    intersect_height = max(0.0, min(box_a.ymax, box_b.ymax) - top_left_y)
    intersection_area = float(intersect_width * intersect_height)

    area1 = (box_a.xmax - box_a.xmin) * (box_a.ymax - box_a.ymin)
    area2 = (box_b.xmax - box_b.xmin) * (box_b.ymax - box_b.ymin)

    union_area = float(area1 + area2 - intersection_area)

    return intersection_area / union_area if union_area > 0.0 else 0.0


def calculate_similarity_matrix(set_a, set_b):
    """Calculates similarity matrix for the specified two sets of boxes.

    :param set_a: First set of boxes
    :param set_b: Second set of boxes
    :return: Similarity matrix
    """

    similarity = np.zeros([len(set_a), len(set_b)], dtype=np.float32)
    for i, _ in enumerate(set_a):
        for j, _ in enumerate(set_b):
            similarity[i, j] = iou(set_a[i], set_b[j])
    return similarity


def load_net(proto_path, weights_path, compute_mode, device_id):
    """Loads network.

    :param proto_path: Path to file with model description
    :param weights_path: Path to file with model  weights
    :param compute_mode: Mode to run network: cpu or gpu
    :param device_id: ID of gpu device
    :return: Loaded network
    """

    def _set_compute_mode():
        """Sets mode to run network: cpu or gpu.
        """

        if compute_mode == 'GPU':
            caffe.set_mode_gpu()
            caffe.set_device(device_id)
        if compute_mode == 'CPU':
            caffe.set_mode_cpu()

    _set_compute_mode()
    net = caffe.Classifier(proto_path, weights_path)

    return net


def extract_detections(image, net, in_blob_name, out_blob_name, min_det_conf=0.1):
    """Runs network to output detected bboxes.

    :param image: Input image
    :param net: Network to run
    :param in_blob_name: Name of network input
    :param out_blob_name: Name of network output
    :param min_det_conf: Threshold to filter invalid detections
    :return: List of detected bboxes
    """

    def _prepare_image(img, size):
        """Transforms input image into network-compatible format.

        :param img: Input image
        :param size: Target size of network input
        :return: Network-compatible input blob
        """

        img = cv2.resize(img, size)
        return img.transpose((2, 0, 1)).astype(np.float32)

    def _is_valid(bbox, height, width):
        """Checks that input bbox is valid according to its size.

        :param bbox: Input bbox
        :param height: Max height
        :param width: Max width
        :return: True of false depending on bbox status
        """

        xmin = max(0, int(round(bbox.xmin * width)))
        ymin = max(0, int(round(bbox.ymin * height)))
        xmax = min(int(round(bbox.xmax * width)), width)
        ymax = min(int(round(bbox.ymax * height)), height)

        out_bbox_h = ymax - ymin
        out_bbox_w = xmax - xmin

        return bbox.det_conf > min_det_conf and out_bbox_h > 0 and out_bbox_w > 0

    in_height, in_width = net.blobs[in_blob_name].data.shape[2:]

    net.blobs['data'].data[...] = _prepare_image(image, (in_width, in_height))

    detections = net.forward()[out_blob_name]
    num_detections = detections.shape[2]

    det_label = detections[0, 0, :, 1].astype(np.int32)
    det_conf = detections[0, 0, :, 2]
    action_conf = detections[0, 0, :, 3]
    det_xmin = detections[0, 0, :, 4]
    det_ymin = detections[0, 0, :, 5]
    det_xmax = detections[0, 0, :, 6]
    det_ymax = detections[0, 0, :, 7]

    all_detections = [BBoxDesc(label=det_label[i],
                               det_conf=det_conf[i],
                               action_conf=action_conf[i],
                               xmin=det_xmin[i], ymin=det_ymin[i],
                               xmax=det_xmax[i], ymax=det_ymax[i])
                      for i in range(num_detections)]

    valid_bbox = [BBoxDesc(label=det.label,
                           det_conf=det.det_conf,
                           action_conf=det.action_conf,
                           xmin=det.xmin, ymin=det.ymin,
                           xmax=det.xmax, ymax=det.ymax)
                  for det in all_detections if _is_valid(det, image.shape[0], image.shape[1])]

    return valid_bbox


def predict_actions(video_path, frame_ids, detection_net, in_det_blob_name, out_det_blob_name):
    """carry out prediction of action for the input video source.

    :param video_path: Path to load video
    :param frame_ids: List of valid frame IDs
    :param detection_net: Network to run
    :param in_det_blob_name: Name of input blob
    :param out_det_blob_name: Name of output blob
    :return: List of predicted actions
    """

    vidcap = cv2.VideoCapture(video_path)

    predicted_detections = {}

    success = True
    frame_id = -1
    pbar = tqdm(total=len(frame_ids), desc='Predicting actions')
    while success:
        success, frame = vidcap.read()
        frame_id += 1

        if success:
            if frame_id not in frame_ids:
                continue

            detections = extract_detections(frame, detection_net, in_det_blob_name, out_det_blob_name)

            predicted_detections[frame_id] = detections
            pbar.update(1)
    pbar.close()

    return predicted_detections


def read_video_size(video_path):
    """Extracts height and width of input video.

    :param video_path: Path to video file
    :return: Height and width of video
    """

    vidcap = cv2.VideoCapture(video_path)

    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('Input video size HxW: {}x{}'.format(height, width))

    return height, width


def match_detections(predicted_data, gt_data, min_iou):
    """Carry out matching between detected and ground truth bboxes.

    :param predicted_data: List of predicted bboxes
    :param gt_data: List of ground truth bboxes
    :param min_iou: Min IoU value to match bboxes
    :return: List of matches
    """

    all_matches = {}
    total_gt_bbox_num = 0
    matched_gt_bbox_num = 0

    frame_ids = list(gt_data)
    for frame_id in tqdm(frame_ids, desc='Matching detections'):
        if frame_id not in predicted_data:
            all_matches[frame_id] = []
            continue

        gt_bboxes = gt_data[frame_id]
        predicted_bboxes = predicted_data[frame_id]

        total_gt_bbox_num += len(gt_bboxes)

        similarity_matrix = calculate_similarity_matrix(gt_bboxes, predicted_bboxes)

        matches = []
        for _ in range(len(gt_bboxes)):
            best_match_pos = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
            best_match_value = similarity_matrix[best_match_pos]

            if best_match_value <= min_iou:
                break

            gt_id = best_match_pos[0]
            predicted_id = best_match_pos[1]

            similarity_matrix[gt_id, :] = 0.0
            similarity_matrix[:, predicted_id] = 0.0

            matches.append((gt_id, predicted_id))
            matched_gt_bbox_num += 1

        all_matches[frame_id] = matches

    print('Matched gt bbox: {} / {} ({:.2f}%)'
          .format(matched_gt_bbox_num, total_gt_bbox_num,
                  100. * float(matched_gt_bbox_num) / float(max(1, total_gt_bbox_num))))

    return all_matches


def calc_confusion_matrix(all_matched_ids, predicted_data, gt_data, num_classes):
    """Calculates confusion matrix.

    :param all_matched_ids: List of matched bboxes
    :param predicted_data: List of predicted bboxes
    :param gt_data: List of ground truth bboxes
    :param num_classes: Number of valid classes
    :return: Confusion matrix
    """

    out_cm = np.zeros([num_classes, num_classes], dtype=np.int32)
    for frame_id in tqdm(all_matched_ids, desc='Evaluating'):
        matched_ids = all_matched_ids[frame_id]
        for match_pair in matched_ids:
            gt_label = gt_data[frame_id][match_pair[0]].label
            pred_label = predicted_data[frame_id][match_pair[1]].label

            if gt_label >= len(VALID_ACTION_NAMES):
                continue

            out_cm[gt_label, pred_label] += 1
    return out_cm


def detection_classagnostic_metrics(all_matched_ids, predicted_data):
    """Calculates detector metrics in case of two-class task.

    :param all_matched_ids: List of matched detections.
    :param predicted_data: List of predicted bboxes
    :return: Detection metrics
    """

    num_detections = int(np.sum([len(bbox_list) for bbox_list in itervalues(predicted_data)]))
    true_positives = np.zeros([num_detections], dtype=np.int32)
    false_positives = np.ones([num_detections], dtype=np.int32)
    scores = np.array([bbox.det_conf for bbox_list in itervalues(predicted_data) for bbox in bbox_list],
                      dtype=np.float32)

    bias = 0
    for frame_id in all_matched_ids:
        matched_ids = all_matched_ids[frame_id]
        predicted_bboxes = predicted_data[frame_id]

        for match_pair in matched_ids:
            pred_local_pos = match_pair[1]

            true_positives[bias + pred_local_pos] = 1
            false_positives[bias + pred_local_pos] = 0

        bias += len(predicted_bboxes)

    return scores, true_positives, false_positives


def detection_classspecific_metrics(all_matched_ids, predicted_data, gt_data, class_id):
    """Calculates detector metrics for the specified class.

    :param all_matched_ids: List of matched detections.
    :param predicted_data: List of predicted bboxes
    :param gt_data: List of ground truth bboxes
    :param class_id: ID of class to calculate
    :return: Detection metrics
    """

    filtered_predicted_pos = {item: [i for i, b in enumerate(l) if b.label == class_id]
                              for item, l in iteritems(predicted_data)}
    filtered_predicted_pos = {item: {i: pos for pos, i in enumerate(ids)}
                              for item, ids in iteritems(filtered_predicted_pos)}

    num_predicted_detections = int(np.sum([len(l) for l in itervalues(filtered_predicted_pos)]))
    true_positives = np.zeros([num_predicted_detections], dtype=np.int32)
    false_positives = np.ones([num_predicted_detections], dtype=np.int32)

    bias = 0
    for frame_id in all_matched_ids:
        matched_ids = all_matched_ids[frame_id]
        predicted_class_positions = filtered_predicted_pos[frame_id]
        gt_bboxes = gt_data[frame_id]

        for match_pair in matched_ids:
            if gt_bboxes[match_pair[0]].label != class_id:
                continue

            if match_pair[1] not in predicted_class_positions:
                continue

            pred_local_pos = predicted_class_positions[match_pair[1]]
            true_positives[bias + pred_local_pos] = 1
            false_positives[bias + pred_local_pos] = 0

        bias += len(predicted_class_positions)

    scores = np.array([b.action_conf for l in itervalues(predicted_data) for b in l if b.label == class_id],
                      dtype=np.float32)

    return scores, true_positives, false_positives


def calc_mr_ap(scores, true_positives, false_positives, num_gt, num_images, fppi_level=0.1):
    """Calculates miss-rate and average precision metrics.

    :param scores: List of confidences
    :param true_positives: List of true positive values
    :param false_positives: List of false positive values
    :param num_gt: total number of ground truth bboxes
    :param num_images: Total number of frames
    :param fppi_level: Threshold of false positive
    :return:
    """

    def _ap(recall, precision):
        """Calculates average precision metric.

        :param recall: List of recall values
        :param precision: List of precision values
        :return: Average precision value
        """

        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        average_precision = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return average_precision

    def _miss_rate(miss_rates, fppis):
        """calculates miss-rate.

        :param miss_rates: List of miss-rates
        :param fppis: List of cumulative false positive values
        :return: Miss-rate value
        """

        position = bisect(fppis, fppi_level)

        left_position = position - 1
        right_position = position if position < len(miss_rates) else left_position

        return 0.5 * (miss_rates[left_position] + miss_rates[right_position])

    if len(true_positives) == 0 or np.sum(true_positives) == 0:
        return 1.0, 0.0

    sorted_ind = np.argsort(-scores)
    fp_sorted = false_positives[sorted_ind]
    tp_sorted = true_positives[sorted_ind]

    fp_cumsum = np.cumsum(fp_sorted)
    tp_cumsum = np.cumsum(tp_sorted)

    ind = len(scores) - np.unique(scores[sorted_ind[::-1]], return_index=True)[1] - 1
    ind = ind[::-1]

    fp_cumsum = fp_cumsum[ind]
    tp_cumsum = tp_cumsum[ind]

    recall = tp_cumsum / float(num_gt)
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
    miss_rates = 1.0 - recall
    fppis = fp_cumsum / float(num_images)

    miss_rate_value = _miss_rate(miss_rates, fppis)
    average_precision_value = _ap(recall, precision)

    return miss_rate_value, average_precision_value


def normalize_confusion_matrix(input_cm):
    """Carry out normalization of rows of input confusion matrix.

    :param input_cm: Input confusion matrix
    :return: Normalized confusion matrix
    """

    assert len(input_cm.shape) == 2
    assert input_cm.shape[0] == input_cm.shape[1]

    row_sums = np.maximum(1, np.sum(input_cm, axis=1, keepdims=True)).astype(np.float32)
    norm_cm = input_cm.astype(np.float32) / row_sums

    return norm_cm


def print_confusion_matrix(input_cm, name, classes):
    """Prints input confusion matrix in human-readable format.

    :param input_cm: Input confusion matrix
    :param name: Name of the printed table
    :param classes: Class names
    """

    assert len(input_cm.shape) == 2
    assert input_cm.shape[0] == input_cm.shape[1]
    assert len(classes) == input_cm.shape[0]

    max_class_name_length = max([len(cl) for cl in classes])

    norm_cm = normalize_confusion_matrix(input_cm)

    print('{} CM:'.format(name))
    for i, class_name in enumerate(classes):
        values = ''
        for j in range(len(classes)):
            values += '{:7.2f} |'.format(norm_cm[i, j] * 100.)
        print('   {0: <{1}}|{2}'.format(class_name, max_class_name_length + 1, values))


def calculate_accuracy(input_cm):
    """Calculates Accuracy metric.

    :param input_cm: Input confusion matrix
    :return: Accuracy value
    """

    assert len(input_cm.shape) == 2
    assert input_cm.shape[0] == input_cm.shape[1]

    base_accuracy = float(np.sum(input_cm.diagonal())) / float(np.maximum(1, np.sum(input_cm)))

    norm_cm = normalize_confusion_matrix(input_cm)
    norm_accuracy = np.mean(norm_cm.diagonal())

    return base_accuracy, norm_accuracy


def read_config(path):
    config = json.load(open(path))
    action_names_map = config['class_names_map']
    valid_action_names = config['valid_class_names']
    return action_names_map, valid_action_names


def main():
    """Calculates action metrics.
    """

    parser = ArgumentParser()
    parser.add_argument('--proto', '-p', type=str, required=True, help='Path to .prototxt')
    parser.add_argument('--weights', '-w', type=str, required=True, help='Path to the .caffemodel')
    parser.add_argument('--tasks', '-t', type=str, required=True, help='Path to video file')
    parser.add_argument('--in_name', default='data', type=str, help='Name of detection input blob')
    parser.add_argument('--out_name', default='detection_out', type=str, help='Name of detection output blob')
    parser.add_argument('--compute_mode', type=str, choices=['CPU', 'GPU'], default='GPU',
                        help='Caffe compute mode: CPU or GPU')
    parser.add_argument('--gpu_id', type=int, default=0, choices=range(8), help='GPU id')
    parser.add_argument('--config', type=str, default='data_config.json', help='')

    args = parser.parse_args()

    assert exists(args.proto)
    assert exists(args.weights)
    assert exists(args.tasks)
    assert exists(args.config)

    action_names_map, valid_action_names = read_config(args.config)

    detection_net = load_net(args.proto, args.weights, args.compute_mode, args.gpu_id)

    num_classes = len(valid_action_names)
    glob_confusion_matrix = np.zeros([num_classes, num_classes], dtype=np.int32)
    glob_scores = np.array([], dtype=np.float32)
    glob_tp = np.array([], dtype=np.int32)
    glob_fp = np.array([], dtype=np.int32)
    glob_num_gt = 0
    glob_num_images = 0

    glob_class_scores = {item: np.array([], dtype=np.float32) for item in range(num_classes)}
    glob_class_tp = {item: np.array([], dtype=np.int32) for item in range(num_classes)}
    glob_class_fp = {item: np.array([], dtype=np.int32) for item in range(num_classes)}
    glob_class_num_gt = {item: 0 for item in range(num_classes)}
    glob_class_num_images = {item: 0 for item in range(num_classes)}

    tasks = parse_tasks(args.tasks)
    for task_id, task in enumerate(tasks):
        task_name = basename(task[0]).split('.')[0]
        print('\nProcessing task {} / {}: \'{}\''.format(task_id + 1, len(tasks), task_name))

        video_size = read_video_size(task[1])
        annotation, _ = load_annotation(task[0], video_size, action_names_map)

        valid_frame_ids = list(annotation)
        predicted_actions = predict_actions(task[1], valid_frame_ids,
                                            detection_net, args.in_name, args.out_name)

        all_matches = match_detections(predicted_actions, annotation, min_iou=0.5)

        local_confusion_matrix = calc_confusion_matrix(all_matches, predicted_actions, annotation, num_classes)
        scores, local_tp, local_fp =\
            detection_classagnostic_metrics(all_matches, predicted_actions)
        num_gt_bboxes = np.sum([len(bbox_list) for bbox_list in itervalues(annotation)])

        task_num_images = len(annotation)

        miss_rate, average_precision = calc_mr_ap(scores, local_tp, local_fp, num_gt_bboxes, task_num_images)
        print('Task Detection Metrics. Class-agnostic AP: {:.2f} miss_rate@0.1: {:.2f}'
              .format(100. * average_precision, 100. * miss_rate))

        glob_confusion_matrix += local_confusion_matrix
        glob_scores = np.concatenate((glob_scores, scores))
        glob_tp = np.concatenate((glob_tp, local_tp))
        glob_fp = np.concatenate((glob_fp, local_fp))
        glob_num_gt += num_gt_bboxes
        glob_num_images += task_num_images

        print('Task Detection Metrics by classes:')
        map_sum = 0.0
        for class_id in range(num_classes):
            class_scores, class_tp, class_fp =\
                detection_classspecific_metrics(all_matches, predicted_actions, annotation, class_id)
            class_num_gt_bboxes = np.sum([len([True for b in bbox_list if b.label == class_id])
                                          for bbox_list in itervalues(annotation)])

            class_gt_image_ids = [img_id for img_id, bbox_list in iteritems(annotation)
                                  if len([True for b in bbox_list if b.label == class_id]) > 0]
            class_pred_image_ids = [img_id for img_id, bbox_list in iteritems(predicted_actions)
                                    if len([True for b in bbox_list if b.label == class_id]) > 0]
            class_num_images = np.sum(len(np.unique(class_gt_image_ids + class_pred_image_ids)))

            class_mr, class_ap = calc_mr_ap(class_scores, class_tp, class_fp, class_num_gt_bboxes, class_num_images)
            print('   {}: AP: {:.2f} miss_rate@0.1: {:.2f}'
                  .format(valid_action_names[class_id], 100. * class_ap, 100. * class_mr))

            map_sum += class_ap

            glob_class_scores[class_id] = np.concatenate((glob_class_scores[class_id], class_scores))
            glob_class_tp[class_id] = np.concatenate((glob_class_tp[class_id], class_tp))
            glob_class_fp[class_id] = np.concatenate((glob_class_fp[class_id], class_fp))
            glob_class_num_gt[class_id] += class_num_gt_bboxes
            glob_class_num_images[class_id] += class_num_images
        map_value = map_sum / float(num_classes)
        print('Task Detection Metrics. mAP: {:.2f}'.format(100. * map_value))

        base_accuracy, norm_accuracy = calculate_accuracy(local_confusion_matrix)
        print('Task Action Accuracy. Base: {:.2f} Norm: {:.2f}'.format(base_accuracy * 100., norm_accuracy * 100.))
        print_confusion_matrix(local_confusion_matrix, 'Task', valid_action_names)

    glob_mr, glob_ap = calc_mr_ap(glob_scores, glob_tp, glob_fp, glob_num_gt, glob_num_images)
    print('\nGlob Detection Metrics. Class-agnostic AP: {:.2f} miss_rate@0.1: {:.2f}'
          .format(100. * glob_ap, 100. * glob_mr))

    print('Glob Detection Metrics by classes:')
    glob_map_sum = 0.0
    for class_id in range(num_classes):
        class_mr, class_ap = calc_mr_ap(glob_class_scores[class_id], glob_class_tp[class_id], glob_class_fp[class_id],
                                        glob_class_num_gt[class_id], glob_class_num_images[class_id])
        print('   {}: AP: {:.2f} miss_rate@0.1: {:.2f}'
              .format(valid_action_names[class_id], 100. * class_ap, 100. * class_mr))

        glob_map_sum += class_ap
    glob_map_value = glob_map_sum / float(num_classes)
    print('Glob Detection Metrics. mAP: {:.2f}'.format(100. * glob_map_value))

    glob_base_accuracy, glob_norm_accuracy = calculate_accuracy(glob_confusion_matrix)
    print('Glob Accuracy. Base: {:.2f} Norm: {:.2f}'.format(glob_base_accuracy * 100., glob_norm_accuracy * 100.))
    print_confusion_matrix(glob_confusion_matrix, 'Glob', valid_action_names)


if __name__ == '__main__':
    main()

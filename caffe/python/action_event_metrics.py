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


# pylint: disable=ungrouped-imports
import os

import json
from argparse import ArgumentParser
from collections import namedtuple
from os import makedirs
from os.path import exists, basename, join

import cv2
import numpy as np
from lxml import etree
from tqdm import tqdm

os.environ['GLOG_minloglevel'] = '2'

# pylint: disable=wrong-import-position
import caffe


BBoxDesc = namedtuple('BBoxDesc', 'id, label, det_conf, action_conf, xmin, ymin, xmax, ymax')
MatchDesc = namedtuple('MatchDesc', 'gt, pred')
Range = namedtuple('Range', 'start, end, label')

IDS_SHIFT_SCALE = 1000000


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


def load_annotation(annotation_path, video_height, video_width, action_names_map):
    """Loads annotation from the specified file.

    :param annotation_path: Path to file with annotation
    :param video_height: Input video height
    :param video_width: Input video width
    :return: Loaded annotation
    """

    tree = etree.parse(annotation_path)
    root = tree.getroot()

    detections_by_frame_id = {}
    ordered_track_id = -1

    for track in tqdm(root, desc='Extracting annotation'):
        if 'label' not in list(track.attrib.keys()) or track.attrib['label'] != 'person':
            continue

        ordered_track_id += 1
        track_id = ordered_track_id if 'id' not in track.attrib else int(track.attrib['id'])

        for bbox in track:
            if len(bbox) < 1:
                continue

            frame_id = int(bbox.attrib['frame'])
            if frame_id <= 0:
                continue

            action_name = None
            for bbox_attr_id in range(len(bbox)):
                attribute_name = bbox[bbox_attr_id].attrib['name']
                if attribute_name != 'action':
                    continue

                action_name = bbox[bbox_attr_id].text

                break

            if action_name is not None and action_name in list(action_names_map.keys()):
                label = action_names_map[action_name]

                xmin = float(bbox.attrib['xtl'])
                ymin = float(bbox.attrib['ytl'])
                xmax = float(bbox.attrib['xbr'])
                ymax = float(bbox.attrib['ybr'])

                bbox_desc = BBoxDesc(id=track_id,
                                     label=label,
                                     det_conf=1.0,
                                     action_conf=1.0,
                                     xmin=xmin / float(video_width),
                                     ymin=ymin / float(video_height),
                                     xmax=xmax / float(video_width),
                                     ymax=ymax / float(video_height))

                detections_by_frame_id[frame_id] = detections_by_frame_id.get(frame_id, []) + [bbox_desc]

    print('Loaded {} annotated frames.'.format(len(detections_by_frame_id)))

    return detections_by_frame_id


def iou(box_a, box_b):
    """ Calculates Intersection over Union (IoU) metric.

    :param box_a: First bbox
    :param box_b: Second bbox
    :return: Scalar value of metric
    """

    intersect_top_left_x = max(box_a.xmin, box_b.xmin)
    intersect_top_left_y = max(box_a.ymin, box_b.ymin)
    intersect_width = max(0.0, min(box_a.xmax, box_b.xmax) - intersect_top_left_x)
    intersect_height = max(0.0, min(box_a.ymax, box_b.ymax) - intersect_top_left_y)

    box_a_area = (box_a.xmax - box_a.xmin) * (box_a.ymax - box_a.ymin)
    box_b_area = (box_b.xmax - box_b.xmin) * (box_b.ymax - box_b.ymin)
    intersect_area = float(intersect_width * intersect_height)

    union_area = float(box_a_area + box_b_area - intersect_area)

    return intersect_area / union_area if union_area > 0.0 else 0.0


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


def extract_detections(image, net, in_blob_name, out_blob_name, min_det_conf, min_action_conf):
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
        return bbox.det_conf > min_det_conf and ymax > ymin and xmax > xmin

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

    all_detections = [BBoxDesc(id=-1,
                               label=det_label[i],
                               det_conf=det_conf[i],
                               action_conf=action_conf[i],
                               xmin=det_xmin[i], ymin=det_ymin[i],
                               xmax=det_xmax[i], ymax=det_ymax[i])
                      for i in range(num_detections)]

    valid_bbox = [BBoxDesc(id=det.id,
                           label=0 if det.label > 0 and det.action_conf < min_action_conf else det.label,
                           det_conf=det.det_conf,
                           action_conf=0.0 if det.label > 0 and det.action_conf < min_action_conf else det.action_conf,
                           xmin=det.xmin, ymin=det.ymin,
                           xmax=det.xmax, ymax=det.ymax)
                  for det in all_detections if _is_valid(det, image.shape[0], image.shape[1])]

    return valid_bbox


def process_video(video_path, frame_ids, detection_net, in_det_blob_name, out_det_blob_name,
                  min_det_conf, min_action_conf):
    """Carry out prediction of action for the input video source.

    :param video_path: Path to load video
    :param frame_ids: List of valid frame IDs
    :param detection_net: Network to run
    :param in_det_blob_name: Name of input blob
    :param out_det_blob_name: Name of output blob
    :param min_det_conf: Min detection confidence
    :param min_action_conf: Min action confidence
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

            detections = extract_detections(frame, detection_net, in_det_blob_name, out_det_blob_name,
                                            min_det_conf, min_action_conf)

            predicted_detections[frame_id] = detections
            pbar.update(1)

        if not success and frame_id == 0:
            success = True

    pbar.close()

    return predicted_detections


def read_video_properties(video_path):
    """Extracts height, width and fps of input video.

    :param video_path: Path to video file
    :return: Height width and fps of video
    """

    vidcap = cv2.VideoCapture(video_path)

    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    print('Input video size (HxW): {} x {}. FPS: {}'.format(height, width, fps))

    return height, width, fps


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

    frame_ids = list(gt_data.keys())
    for frame_id in tqdm(frame_ids, desc='Matching detections'):
        if frame_id not in list(predicted_data.keys()):
            all_matches[frame_id] = []
            continue

        gt_bboxes = gt_data[frame_id]
        predicted_bboxes = predicted_data[frame_id]

        total_gt_bbox_num += len(gt_bboxes)

        sorted_predicted_bboxes = [(i, b) for i, b in enumerate(predicted_bboxes)]
        sorted_predicted_bboxes.sort(key=lambda tup: tup[1].det_conf, reverse=True)

        matches = []
        visited_gt = np.zeros(len(gt_bboxes), dtype=np.bool)
        for i in range(len(sorted_predicted_bboxes)):
            predicted_id = sorted_predicted_bboxes[i][0]
            predicted_bbox = sorted_predicted_bboxes[i][1]

            best_overlap = 0.0
            best_gt_id = -1
            for gt_id in range(len(gt_bboxes)):
                if visited_gt[gt_id]:
                    continue

                overlap_value = iou(predicted_bbox, gt_bboxes[gt_id])
                if overlap_value > best_overlap:
                    best_overlap = overlap_value
                    best_gt_id = gt_id

            if best_gt_id >= 0 and best_overlap > min_iou:
                visited_gt[best_gt_id] = True

                matches.append((best_gt_id, predicted_id))
                matched_gt_bbox_num += 1

                if len(matches) >= len(gt_bboxes):
                    break

        all_matches[frame_id] = matches

    print('Matched gt bbox: {} / {} ({:.2f}%)'
          .format(matched_gt_bbox_num, total_gt_bbox_num,
                  100. * float(matched_gt_bbox_num) / float(max(1, total_gt_bbox_num))))

    return all_matches


def split_to_tracks(gt_data):
    """Splits data to tracks according ID.

    :param gt_data: Input data
    :return: List of tracks
    """

    tracks = {}
    for frame_id in tqdm(gt_data, desc='Splitting GT'):
        gt_frame_data = gt_data[frame_id]
        for bbox in gt_frame_data:
            track_id = bbox.id

            new_match = MatchDesc(bbox, None)

            if track_id not in tracks:
                tracks[track_id] = {frame_id: new_match}
            else:
                tracks[track_id][frame_id] = new_match

    return tracks


def add_matched_predictions(tracks, all_matched_ids, predicted_data, gt_data):
    """Adds matched predicted events to the input tracks.

    :param tracks: Input tracks
    :param all_matched_ids: List of matches
    :param predicted_data: Predicted data
    :param gt_data: ground-truth data
    :return: Updated list of tracks
    """

    for frame_id in tqdm(list(all_matched_ids.keys()), desc='Splitting Predictions'):
        gt_frame_data = gt_data[frame_id]
        predicted_frame_data = predicted_data[frame_id]
        matched_ids = all_matched_ids[frame_id]

        for match_pair in matched_ids:
            track_id = gt_frame_data[match_pair[0]].id
            predicted_bbox = predicted_frame_data[match_pair[1]]

            new_match = tracks[track_id][frame_id]._replace(pred=predicted_bbox)
            tracks[track_id][frame_id] = new_match

    return tracks


def extract_events(frame_events, window_size, min_length, frame_limits):
    """Merges input frame-based tracks to event-based ones.

    :param frame_events: Input tracks
    :param window_size: Size of smoothing window
    :param min_length: Min duration of event
    :param frame_limits: Start and end frame ID
    :return: List of event-based tracks
    """

    def _smooth(input_events):
        """Merge frames into the events of the same action.

        :param input_events: Frame-based actions
        :return: List of events
        """

        out_events = []

        if len(input_events) > 0:
            last_range = Range(input_events[0][0], input_events[0][0] + 1, input_events[0][1])
            for i in range(1, len(input_events)):
                if last_range.end + window_size - 1 >= input_events[i][0] and last_range.label == input_events[i][1]:
                    last_range = last_range._replace(end=input_events[i][0] + 1)
                else:
                    out_events.append(last_range)

                    last_range = Range(input_events[i][0], input_events[i][0] + 1, input_events[i][1])

            out_events.append(last_range)

        return out_events

    def _filter(input_events):
        """Filters too short events.

        :param input_events: List of events
        :return: Filtered list of events
        """

        return [e for e in input_events if e.end - e.start >= min_length]

    def _extrapolate(input_events):
        """Expands time limits of the input events to the specified one.

        :param input_events: List of events
        :return: Expanded list of events
        """

        out_events = []

        if len(input_events) == 1:
            out_events = [Range(frame_limits[0], frame_limits[1], input_events[0].label)]
        elif len(input_events) > 1:
            first_event = input_events[0]._replace(start=frame_limits[0])
            last_event = input_events[-1]._replace(end=frame_limits[1])
            out_events = [first_event] + input_events[1:-1] + [last_event]

        return out_events

    def _interpolate(input_events):
        """Fills event-free ranges by interpolating neighbouring events.

        :param input_events: List of events
        :return: Filled list of events
        """

        out_events = []

        if len(input_events) > 0:
            last_event = input_events[0]
            for event_id in range(1, len(input_events)):
                cur_event = input_events[event_id]

                middle_point = int(0.5 * (last_event.end + cur_event.start))

                last_event = last_event._replace(end=middle_point)
                cur_event = cur_event._replace(start=middle_point)

                out_events.append(last_event)
                last_event = cur_event
            out_events.append(last_event)

        return out_events

    def _merge(input_events):
        """Merges consecutive events of the same action.

        :param input_events: List of events
        :return: Merged list of events
        """

        out_events = []

        if len(input_events) > 0:
            last_event = input_events[0]
            for cur_event in input_events[1:]:
                if last_event.end == cur_event.start and last_event.label == cur_event.label:
                    last_event = last_event._replace(end=cur_event.end)
                else:
                    out_events.append(last_event)
                    last_event = cur_event
            out_events.append(last_event)

        return out_events

    events = _smooth(frame_events)
    events = _filter(events)
    events = _extrapolate(events)
    events = _interpolate(events)
    events = _merge(events)

    return events


def match_events(gt_events, pred_events):
    """Carry out matching between two input sets of events.

    :param gt_events: Input ground-truth events
    :param pred_events: Input predicted events
    :return: List of matched events
    """

    num_gt = len(gt_events)
    num_pred = len(pred_events)
    if num_gt == 0 or num_pred == 0:
        return []

    matches = []
    for pred_id in range(len(pred_events)):
        best_overlap_value = 0
        best_gt_id = -1
        for gt_id in range(len(gt_events)):
            intersect_start = np.maximum(gt_events[gt_id].start, pred_events[pred_id].start)
            intersect_end = np.minimum(gt_events[gt_id].end, pred_events[pred_id].end)

            overlap = np.maximum(0, intersect_end - intersect_start)
            overlap = 0 if gt_events[gt_id].label != pred_events[pred_id].label else overlap

            if overlap > best_overlap_value:
                best_overlap_value = overlap
                best_gt_id = gt_id

        if best_overlap_value > 0 and best_gt_id >= 0:
            matches.append((best_gt_id, pred_id))

    return matches


def process_tracks(all_tracks, window_size, min_length, ignore_class_id):
    """Carry out smoothing of the input tracks

    :param all_tracks: Input tracks
    :param window_size: Size of smooth window
    :param min_length: Min duration of event
    :return: List of smoothed tracks
    """

    out_tracks = {}
    for track_id in tqdm(list(all_tracks.keys()), desc='Extracting events'):
        track = all_tracks[track_id]

        frame_ids = list(track)
        frame_ids.sort()
        frame_id_limits = np.min(frame_ids), np.max(frame_ids) + 1

        gt_frame_events = [(fi, track[fi].gt.label) for fi in frame_ids if track[fi].gt.label != ignore_class_id]
        pred_frame_events = [(fi, track[fi].pred.label) for fi in frame_ids if track[fi].pred is not None]

        # skip unmatched track
        if len(gt_frame_events) == 0 or len(pred_frame_events) == 0:
            continue

        gt_events = extract_events(gt_frame_events, window_size, min_length, frame_id_limits)
        pred_events = extract_events(pred_frame_events, window_size, min_length, frame_id_limits)

        out_tracks[track_id] = gt_events, pred_events

    return out_tracks


def calculate_metrics(all_tracks):
    """Calculates Precision and Recall metrics.

    :param all_tracks: Input mathed events
    :return: Precision and Recall scalar values
    """

    total_num_pred_events = 0
    total_num_valid_pred_events = 0
    total_num_gt_events = 0
    total_num_valid_gt_events = 0

    for track_id in all_tracks:
        gt_events, pred_events = all_tracks[track_id]

        matches = match_events(gt_events, pred_events)

        total_num_pred_events += len(pred_events)
        total_num_gt_events += len(gt_events)

        if len(matches) > 0:
            matched_gt = np.zeros([len(gt_events)], dtype=np.bool)
            matched_pred = np.zeros([len(pred_events)], dtype=np.bool)
            for match in matches:
                matched_gt[match[0]] = True
                matched_pred[match[1]] = True

            total_num_valid_pred_events += np.sum(matched_pred)
            total_num_valid_gt_events += np.sum(matched_gt)

    precision = float(total_num_valid_pred_events) / float(total_num_pred_events)\
        if total_num_pred_events > 0 else 0.0
    recall = float(total_num_valid_gt_events) / float(total_num_gt_events)\
        if total_num_gt_events > 0 else 0.0

    return precision, recall


def save_tracks(input_tracks, out_filename):
    """Saves smoothed tracks to the specified file.

    :param input_tracks: List of input tracks
    :param out_filename: File to save tracks
    """

    def _convert_track(input_track):
        """Converts event from internal representation to dict format

        :param input_track: List of events
        :return: Event as dict
        """

        return [event._asdict() for event in input_track]

    out_tracks = {}
    for track_id in input_tracks:
        gt_events, pred_events = input_tracks[track_id]

        converted_gt_events = _convert_track(gt_events)
        converted_pred_events = _convert_track(pred_events)

        out_tracks[track_id] = {'gt': converted_gt_events,
                                'pred': converted_pred_events}

    with open(out_filename, 'w') as outfile:
        json.dump(out_tracks, outfile)


def read_config(path):
    config = json.load(open(path))
    action_names_map = config['class_names_map']
    valid_action_names = config['valid_class_names']
    ignore_class_id = config['ignore_class_id']
    return action_names_map, valid_action_names, ignore_class_id


def main():
    """Calculates event-based action metrics.
    """

    parser = ArgumentParser()
    parser.add_argument('--proto', '-p', type=str, required=True, help='Path to .prototxt')
    parser.add_argument('--weights', '-w', type=str, required=True, help='Path to the .caffemodel')
    parser.add_argument('--tasks', '-t', type=str, required=True, help='Path to video file')
    parser.add_argument('--in_name', default='data', type=str, help='Name of detection input blob')
    parser.add_argument('--out_name', default='detection_out', type=str, help='Name of detection output blob')
    parser.add_argument('--compute_mode', type=str, choices=['CPU', 'GPU'], default='GPU',
                        help='Caffe compute mode: CPU or GPU')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, choices=list(range(8)), help='GPU id')
    parser.add_argument('--min_detection_conf', '-dc', type=float, default=0.4, help='Min detection conf')
    parser.add_argument('--min_action_conf', '-ac', type=float, default=0.7, help='Min action conf')
    parser.add_argument('--min_action_length', type=float, default=1.0, help='Min action duration (s)')
    parser.add_argument('--window_size', type=float, default=1.0, help='Smooth window size (s)')
    parser.add_argument('--out_dir', type=str, default='', help='Path to save smoothed annotation')
    parser.add_argument('--config', type=str, default='data_config.json', help='')
    args = parser.parse_args()

    assert exists(args.proto)
    assert exists(args.weights)
    assert exists(args.tasks)
    assert exists(args.config)

    action_names_map, valid_action_names, ignore_class_id = read_config(args.config)

    if args.out_dir is not None and args.out_dir != '':
        if not exists(args.out_dir):
            makedirs(args.out_dir)

    detection_net = load_net(args.proto, args.weights, args.compute_mode, args.gpu_id)

    tasks = parse_tasks(args.tasks)
    for task_id, task in enumerate(tasks):
        task_name = basename(task[0]).split('.')[0]
        print('\nProcessing task {} / {}: \'{}\''.format(task_id + 1, len(tasks), task_name))

        video_height, video_width, video_fps = read_video_properties(task[1])
        annotation = load_annotation(task[0], video_height, video_width, action_names_map)

        valid_frame_ids = list(annotation.keys())
        predicted_actions = process_video(task[1], valid_frame_ids, detection_net, args.in_name, args.out_name,
                                          min_det_conf=args.min_detection_conf, min_action_conf=args.min_action_conf)

        all_matches = match_detections(predicted_actions, annotation, min_iou=0.5)

        tracks = split_to_tracks(annotation)
        print('Found {} tracks.'.format(len(tracks)))

        tracks = add_matched_predictions(tracks, all_matches, predicted_actions, annotation)

        window_size = np.maximum(1, int(args.window_size * video_fps))
        min_action_length = np.maximum(1, int(args.min_action_length * video_fps))
        track_events = process_tracks(tracks, window_size, min_action_length, ignore_class_id)

        print('window_size: {}'.format(window_size))
        print('min_action_length: {}'.format(min_action_length))

        precision, recall = calculate_metrics(track_events)
        print('\nPrecision: {:.3f}%   Recall: {:.3f}%'.format(1e2 * precision, 1e2 * recall))

        if exists(args.out_dir):
            out_path = join(args.out_dir, 'tracks_{}.json'.format(task_name))
            save_tracks(track_events, out_path)


if __name__ == '__main__':
    main()

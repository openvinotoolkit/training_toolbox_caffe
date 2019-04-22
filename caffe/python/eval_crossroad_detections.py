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


import argparse
from copy import deepcopy

import os.path as osp
import glog as logging
import numpy as np
from six import iteritems
from tqdm import tqdm
import json
import cv2

from utils.object_detection_evaluation import DetectionEvaluation
from utils.image_grabber import ImageGrabber

#pylint: disable=invalid-name
image_provider = ImageGrabber()
num_classes = 3
ignored_id = -2
valid_labels = ['person', 'vehicle', 'non-vehicle']
labels_to_class_idx = {'__background__': -1,
                       'ignored': ignored_id,
                       'person': 0,
                       'vehicle': 1,
                       'non-vehicle': 2}


def parse_args():
    """ Parse input parameters
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('ground_truth_file_path')
    parser.add_argument('detections_file_path')
    parser.add_argument('--remap', dest='classes_remap', choices=('all', 'pascal', 'coco'), default='all')
    return parser.parse_args()


#pylint: disable=consider-using-enumerate
def prepare_ground_truth(ground_truth):
    """ Prepare ground truth detections
    """
    for gt in ground_truth:
        to_delete = []
        objects = gt['objects']
        for i in range(len(objects)):
            obj = objects[i]
            if len(obj) == 0:
                to_delete.append(i)
                continue
            # If object has label 'ignored', add the same box
            # for each valid object label and set 'ignored' flag to True.
            if obj['label'] == 'ignored':
                to_delete.append(i)
                for j in range(num_classes):
                    new_obj = deepcopy(obj)
                    new_obj['label'] = valid_labels[j]
                    new_obj['ignored'] = True
                    objects.append(new_obj)
            else:
                obj['difficult'] = obj.get('difficult', False) or obj.get('occluded', False)
                if 'occluded' in obj:
                    del obj['occluded']
                obj['ignored'] = False
        for i in reversed(sorted(to_delete)):
            del objects[i]
    return None

def add_detections(evaluator,
                   detections,
                   ground_truth,
                   verbose=True):
    """ Add found detections to evaluator
    """
    detections_image_paths_list = np.asarray([osp.basename(d['image']) for d in detections])
    ground_truth_image_paths_list = np.asarray([osp.basename(gt['image']) for gt in ground_truth])
    assert np.array_equal(detections_image_paths_list, ground_truth_image_paths_list)

    for image_id, (det, gt) in tqdm(enumerate(zip(detections, ground_truth)),
                                    desc='for every image', disable=not verbose,
                                    total=len(detections)):
        bboxes = []
        labels = []
        ignored_mask = []
        for object_gt in gt['objects']:
            bboxes.append(object_gt['bbox'])
            labels.append(labels_to_class_idx.get(object_gt['label'], -1))
            ignored_mask.append(object_gt['ignored'])
        if len(bboxes) > 0:
            bboxes = np.asarray(bboxes, dtype=np.float32)
            bboxes = bboxes[:, (1, 0, 3, 2)]
            labels = np.asarray(labels, dtype=np.int32)
            ignored_mask = np.asarray(ignored_mask, dtype=bool)
        else:
            bboxes = np.empty((0, 4), dtype=np.float32)
            labels = np.zeros(0, dtype=np.int32)
            ignored_mask = np.ones(0, dtype=bool)

        evaluator.set_ground_truth(image_id, bboxes, labels, ignored_mask)

        bboxes = []
        labels = []
        scores = []
        for object_pred in det['objects']:
            bboxes.append(object_pred['bbox'])
            labels.append(labels_to_class_idx.get(object_pred['label'], -1))
            scores.append(object_pred['score'])
        if len(bboxes) > 0:
            bboxes = np.asarray(bboxes, dtype=np.float32)
            bboxes = bboxes[:, (1, 0, 3, 2)]
            labels = np.asarray(labels, dtype=np.int32)
            scores = np.asarray(scores, dtype=np.float32)
        else:
            bboxes = np.empty((0, 4), dtype=np.float32)
            labels = np.zeros(0, dtype=np.int32)
            scores = np.ones(0, dtype=np.float32)

        evaluator.set_detections(image_id, bboxes, scores, labels)
    return None


def set_relative_area_filter(evaluator, ground_truth, area_range=(0, np.inf)):
    """ Set relative area filter
    """
    for image_id, gt in enumerate(ground_truth):
        # FIXME. This is a workaround for images without GT objects (?) #pylint: disable=fixme
        if evaluator.dataset[image_id].ground_truth_boxes is None:
            continue
        gt_of_interest = np.ones(len(gt['objects']), dtype=np.bool)
        if 'image_size' not in gt:
            path = gt['image']
            image = image_provider.get_image(path)
            if image is None or np.any(image.shape == 0) or len(image.shape) < 2:
                logging.error('Invalid image "{}"'.format(path))
                raise ValueError('Failed to load image')
            image_size = image.shape[:2]
            gt['image_size'] = image_size
        image_size = gt['image_size']
        for i, obj in enumerate(gt['objects']):
            bbox = obj['bbox']
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            area = np.sqrt(float(w * h) / image_size[0] / image_size[1])
            gt_of_interest[i] = (area_range[0] <= area <= area_range[1]) and (not obj.get('difficult', False))
        evaluator.dataset[image_id].set_filter(np.asarray(gt_of_interest))


def evaluate_detections(evaluator):
    """ Evaluate detections
    """
    ap_per_class, mean_ap = evaluator.evaluate()[:2]
    metrics = {'mAP@{}IOU'.format(0.5): mean_ap}
    for class_idx, label in enumerate(('person', 'vehicle', 'non-vehicle')):
        metrics['mAP@{}IOU {}'.format(0.5, label)] = ap_per_class[class_idx]
    return metrics


#pylint: disable=super-init-not-called
class LabelMapper(object):
    """ LabelMapper class
    """
    def __init__(self):
        pass

    @staticmethod
    def all_classes_map():
        """ All classes mapping
        """
        import collections

        class IdentityDict(collections.MutableMapping):
            """ IdentityDict class
            """
            def __init__(self, *args, **kwargs):
                self.store = dict()
                self.update(dict(*args, **kwargs))

            def __getitem__(self, key):
                if key not in self.store:
                    self.store[key] = key
                return self.store[key]

            def __setitem__(self, key, value):
                self.store[key] = value

            def __delitem__(self, key):
                del self.store[key]

            def __iter__(self):
                return iter(self.store)

            def __len__(self):
                return len(self.store)

        return IdentityDict()

    @staticmethod
    def pascal_classes_map():
        """ Mapping of Pascal classes
        """
        relevant_classes = {'__background__': '__background__',
                            'person': 'person',
                            'car': 'vehicle',
                            'bus': 'vehicle',
                            'bicycle': 'non-vehicle',
                            'motorbike': 'non-vehicle',
                            'train': 'vehicle'}
        return relevant_classes

    @staticmethod
    def coco_classes_map():
        """ Mapping of Coco classes
        """
        relevant_classes = {'__background__': '__background__',
                            'person': 'person',
                            'car': 'vehicle',
                            'bus': 'vehicle',
                            'truck': 'vehicle',
                            'bicycle': 'non-vehicle',
                            'motorcycle': 'non-vehicle',
                            'train': 'vehicle'}
        return relevant_classes

    @staticmethod
    def factory(mapping_type):
        """ Mapping factory
        """
        if mapping_type == 'pascal':
            return LabelMapper.pascal_classes_map()
        elif mapping_type == 'coco':
            return LabelMapper.coco_classes_map()
        elif mapping_type == 'all':
            return LabelMapper.all_classes_map()
        else:
            raise ValueError('Unknown mapping type {}'.format(mapping_type))

def main():
    """ Evaluate found detections
    """
    args = parse_args()
    logging.info('loading ground truth from "{}"...'.format(args.ground_truth_file_path))
    ground_truth = json.load(open(args.ground_truth_file_path, 'r'))
    logging.info('loading detections from "{}"...'.format(args.detections_file_path))
    detections = json.load(open(args.detections_file_path, 'r'))

    logging.info('preparing ground truth...')
    prepare_ground_truth(ground_truth)

    print('len det', len(detections))
    print('len gt ', len(ground_truth))

    logging.info('remapping detection labels and doing NMS...')
    label_mapping = LabelMapper.factory(args.classes_remap)
    for image_detections in detections:
        for detected_object in image_detections['objects']:
            detected_object['label'] = label_mapping[detected_object['label']]

    paths_to_images = [d['image'] for d in detections]

    logging.info('computing metrics...')
    evaluator = DetectionEvaluation(3, 0.5,
                                    prefer_regular_ground_truth=True,
                                    allow_multiple_matches_per_ignored=True)
    add_detections(evaluator, detections, ground_truth)

    from itertools import product
    summary_header = product(['Reasonable', 'S', 'M', 'L'], ['', '-non-vehicle', '-person', '-vehicle'])
    summary_header = [''.join([x[0], x[1]]) for x in summary_header]
    summary = []

    print('REASONABLE')
    full_hd_norm = np.sqrt(1920.0 * 1080.0)
    set_relative_area_filter(evaluator, ground_truth, area_range=(50.0 / full_hd_norm, np.inf))
    metrics = evaluate_detections(evaluator)
    for key, value in sorted(iteritems(metrics)):
        summary.append(value)
        print('\t{:>10}: {:.2%}'.format(key, value))

    print('XS')
    set_relative_area_filter(evaluator, ground_truth, area_range=(0, 50.0 / full_hd_norm))
    metrics = evaluate_detections(evaluator)
    for key, value in sorted(iteritems(metrics)):
        print('\t{:>10}: {:.2%}'.format(key, value))

    print('S')
    set_relative_area_filter(evaluator, ground_truth, area_range=(50.0 / full_hd_norm, 75.0 / full_hd_norm))
    metrics = evaluate_detections(evaluator)
    for key, value in sorted(iteritems(metrics)):
        summary.append(value)
        print('\t{:>10}: {:.2%}'.format(key, value))

    print('M')
    set_relative_area_filter(evaluator, ground_truth, area_range=(75.0 / full_hd_norm, 125.0 / full_hd_norm))
    metrics = evaluate_detections(evaluator)
    for key, value in sorted(iteritems(metrics)):
        summary.append(value)
        print('\t{:>10}: {:.2%}'.format(key, value))

    print('L')
    set_relative_area_filter(evaluator, ground_truth, area_range=(125.0 / full_hd_norm, np.inf))
    metrics = evaluate_detections(evaluator)
    for key, value in sorted(iteritems(metrics)):
        summary.append(value)
        print('\t{:>10}: {:.2%}'.format(key, value))

    print(','.join(summary_header))
    print(','.join(['{:.2f}'.format(100 * x) for x in summary]))

if __name__ == "__main__":
    main()

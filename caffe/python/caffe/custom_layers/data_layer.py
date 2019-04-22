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

import time
import traceback
import signal
import json
from builtins import range
from os import listdir
from os.path import exists, basename, isfile, join, dirname
from collections import namedtuple
from multiprocessing import Process, Queue

import cv2
import numpy as np
from lxml import etree
from six import itervalues
from scipy.ndimage.filters import gaussian_filter

from caffe._caffe import log as LOG
from caffe._caffe import Layer as BaseLayer

BBox = namedtuple('BBox', 'track_id, class_id, xmin, ymin, xmax, ymax, occluded')
FrameDesc = namedtuple('FrameDesc', 'path, objects')
TransformParams = namedtuple('TransformParams', 'crop, support_bbox, aspect_ratio, mirror,'
                                                'expand, expand_ratio, expand_shift, expand_fill')
IDS_SHIFT_SCALE = 1000000
VALID_EXTENSIONS = ['png', 'jpg']


class SampleDataFromDisk(object):
    """Auxiliary class to handle the list of valid images and to provide
       the functionality to load them.
    """

    def __init__(self, task_path, ignore_occluded, class_names_map, valid_class_ids, ignore_class_id, use_attribute):
        """Seeks valid images and according annotations files.

        :param task_path: Path to file with tasks to load data
        :param ignore_occluded: Whether to ignore occluded detections
        :param class_names_map: Dictionary to map class names to ID
        :param valid_class_ids: List of valid class IDs
        :param ignore_class_id: ID of ignored class
        """

        self._class_names_map = class_names_map
        self._valid_class_ids = valid_class_ids
        self._ignore_class_id = ignore_class_id

        tasks = self._parse_tasks(task_path)
        LOG('Found {} tasks:'.format(len(tasks)))
        for task_id, task in enumerate(tasks):
            LOG('   {}: {}'.format(task_id, task[0]))

        total_frames = 0
        total_objects = 0

        self._all_frames = []
        self._class_queues = {i: [] for i in self._valid_class_ids if i != self._ignore_class_id}
        glob_class_counts = {i: 0 for i in self._valid_class_ids if i != self._ignore_class_id}

        for task_id, task in enumerate(tasks):
            LOG('Loading task {}...'.format(task[0]))

            annotation_path = task[0]
            images_dir = task[1]
            video_resolution = task[2]

            image_paths = self._parse_images(images_dir)
            if len(image_paths) == 0:
                continue

            id_shift = (task_id + 1) * IDS_SHIFT_SCALE
            annotation = self._read_annotation(annotation_path, video_resolution, ignore_occluded,
                                               self._class_names_map, self._valid_class_ids, id_shift, use_attribute)

            for frame_id in annotation:
                if frame_id not in image_paths:
                    continue

                image_path = image_paths[frame_id]
                gt_objects = annotation[frame_id]
                if len(gt_objects) == 0:
                    continue

                # Skip images without annotated objectes
                if not sum([gt_o.class_id != self._ignore_class_id for gt_o in gt_objects]):
                    continue

                self._all_frames.append(FrameDesc(path=image_path, objects=gt_objects))
                frame_glob_id = len(self._all_frames) - 1

                local_class_counts = {i: 0 for i in self._valid_class_ids if i != self._ignore_class_id}
                for gt_object in gt_objects:
                    class_id = gt_object.class_id
                    if class_id != self._ignore_class_id:
                        local_class_counts[class_id] += 1
                        glob_class_counts[class_id] += 1

                for class_id in local_class_counts:
                    if local_class_counts[class_id] > 0:
                        self._class_queues[class_id].append(frame_glob_id)

                total_frames += 1
                total_objects += len(gt_objects)

        LOG('DataLayer stats: loaded {} frames with {} objects.'.format(total_frames, total_objects))
        self._print_stat(glob_class_counts, self._class_queues, self._ignore_class_id)

        for class_id in self._class_queues:
            if class_id == ignore_class_id:
                continue
            if len(self._class_queues[class_id]) == 0:
                raise Exception('Cannot find frames with {} class id'.format(class_id))

    @staticmethod
    def _print_stat(labels, class_queues, ignore_class_id):
        """Prints statistics of loaded data.

        :param labels: List of loaded labels
        :param class_queues: Lists of frames with specific class
        :param ignore_class_id: ID of ignored class
        """

        total_num = np.sum([labels[label] for label in labels if label != ignore_class_id])
        if total_num == 0:
            LOG('No labels')
            return

        factor = 100. / float(total_num)

        LOG('Labels:')
        for label in labels:
            if label == ignore_class_id:
                LOG('   {}: {:06} - ignored'.format(label, labels[label]))
            else:
                LOG('   {}: {:06} ({:.2f}%) - {:06} frames'
                    .format(label, labels[label], factor * float(labels[label]), len(class_queues[label])))

    @staticmethod
    def _parse_tasks(file_path):
        """Parse the file with tasks for data loading. Each task is presented by row:
               "annotation_path img_height,img_width directory_with_frames".

        :param file_path: Path to file with tasks
        :return: List of data loading tasks
        """

        tasks = []
        data_dir = dirname(file_path)

        with open(file_path, 'r') as file_stream:
            for line in file_stream:
                if line.endswith('\n'):
                    line = line[:-len('\n')]

                if len(line) == 0:
                    continue

                annotation_path, video_resolution, images_dir = line.split(' ')
                annotation_path = join(data_dir, annotation_path)
                images_dir = join(data_dir, images_dir)
                video_resolution = [int(x) for x in video_resolution.split(',')]

                if not exists(annotation_path) or not exists(images_dir):
                    continue

                tasks.append((annotation_path, images_dir, video_resolution))

        return tasks

    @staticmethod
    def _parse_images(images_dir):
        """Parse image paths for the specified directory.

        :param images_dir: Directory path for search
        :return: List of found image paths
        """

        all_files = [join(images_dir, f) for f in listdir(images_dir) if isfile(join(images_dir, f))]

        data = {}
        for file_path in all_files:
            file_name, extension = basename(file_path).split('.')
            if extension not in VALID_EXTENSIONS:
                continue

            frame_id = int(file_name.split('_')[-1])
            if frame_id < 0:
                continue

            data[frame_id] = file_path

        return data

    @staticmethod
    def _read_annotation(annotation_path, image_size, ignore_occluded,
                         class_names_map, valid_class_ids, id_shift, use_attribute):
        """Loads annotation file by the specified path and normalizes bbox coordinates.

        :param annotation_path: Path of annotation file
        :param image_size: Source image size (needed to normalize bbox)
        :param ignore_occluded: Whether to load occluded bboxes
        :param class_names_map: Dictionary to map class names to ID
        :param valid_class_ids: List of valid class IDs
        :param id_shift: Shift for IDs to preserve unique property
        :return: List of detections
        """

        tree = etree.parse(annotation_path)
        root = tree.getroot()

        detections_by_frame_id = {}
        for track in root:
            # TODO
            if 'label' not in track.attrib or track.attrib['label'] != 'person':
                continue

            track_id = int(track.attrib['id'])
            assert track_id < IDS_SHIFT_SCALE, 'Invalid ID: {}'.format(track_id)

            for bbox in track:
                if len(bbox) < 1:
                    continue

                frame_id = int(bbox.attrib['frame'])

                class_name = None
                if use_attribute != "":
                    for bbox_attr_id, _ in enumerate(bbox):
                        attribute_name = bbox[bbox_attr_id].attrib['name']
                        if attribute_name != use_attribute:
                            continue
                        class_name = bbox[bbox_attr_id].text
                else:
                    class_name = track.attrib['label']

                if class_name is None or class_name not in class_names_map:
                    continue

                class_name_id = class_names_map[class_name]
                if class_name_id not in valid_class_ids:
                    continue

                is_occluded = bbox.attrib['occluded'] == '1'
                if ignore_occluded and is_occluded:
                    class_name_id = class_names_map['__undefined__']

                bbox_desc = BBox(track_id=track_id + id_shift,
                                 class_id=class_name_id,
                                 occluded=is_occluded,
                                 xmin=float(bbox.attrib['xtl']) / float(image_size[1]),
                                 ymin=float(bbox.attrib['ytl']) / float(image_size[0]),
                                 xmax=float(bbox.attrib['xbr']) / float(image_size[1]),
                                 ymax=float(bbox.attrib['ybr']) / float(image_size[0]))
                detections_by_frame_id[frame_id] = detections_by_frame_id.get(frame_id, []) + [bbox_desc]

        return detections_by_frame_id

    def get_class_frame_ids(self):
        """Returns frame IDs for each class.

        :return: Frame IDs.
        """

        return self._class_queues

    def get_frame_with_annotation(self, frame_id):
        """Loads image and its annotation.

        :param frame_id: ID of image to load
        :return: Tuple of image and its annotation
        """

        desc = self._all_frames[frame_id]

        image = cv2.imread(desc.path)
        objects = desc.objects

        return image, objects


class DataLayer(BaseLayer):
    """Layer to provide generation of batch of training samples with data
       loading from disk in parallel mode.
    """

    @staticmethod
    def _image_to_blob(img, img_width, img_height):
        """Transforms to net-compatible format of blob: [channels, height, width].

        :param img: Input image
        :param img_width: Target image height
        :param img_height: Target image width
        :return: Final image blob
        """

        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        blob = img.astype(np.float32).transpose((2, 0, 1))
        return blob

    @staticmethod
    def _objects_to_blob(item_id, objects):
        """Transforms annotation to net-compatible format of blob.

        :param item_id:
        :param objects:
        :return: Final annotation blob
        """

        objects_annotation = []
        for bbox in objects:
            object_desc = [float(x) for x in [item_id, bbox.class_id, bbox.track_id,
                                              bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, 0]]
            objects_annotation.append(object_desc)
        return objects_annotation

    def _sample_params(self, image, objects_list):
        """Samples random parameters for the next image transformers.

        :param image: Input image
        :param objects_list: Annotation list
        :return: List of sampled parameters
        """

        def _rand_bbox():
            """Samples random bounding box.

            :return: Bounding box
            """
            xmin = np.random.uniform(0.0, 1.0)
            ymin = np.random.uniform(0.0, 1.0)

            xmax = np.random.uniform(xmin, 1.0)
            ymax = np.random.uniform(ymin, 1.0)

            return BBox(track_id=-1, class_id=-1,
                        xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                        occluded=False)

        def _find_limits(bbox_set):
            """Find global bounding box of the specified set of bboxes.

            :param bbox_set: List of input bboxes
            :return: Final bounding box
            """

            return BBox(track_id=-1, class_id=-1,
                        xmin=np.min([b.xmin for b in bbox_set]),
                        ymin=np.min([b.ymin for b in bbox_set]),
                        xmax=np.max([b.xmax for b in bbox_set]),
                        ymax=np.max([b.ymax for b in bbox_set]),
                        occluded=False)

        def _bbox_dist(bbox_a, bbox_b):
            center_a = (0.5 * (bbox_a.xmin + bbox_a.xmax), 0.5 * (bbox_a.ymin + bbox_a.ymax))
            center_b = (0.5 * (bbox_b.xmin + bbox_b.xmax), 0.5 * (bbox_b.ymin + bbox_b.ymax))
            return np.square(center_a[0] - center_b[0]) + np.square(center_a[1] - center_b[1])

        class_prob = np.random.uniform(0.0, 1.0)
        do_expand = self.spatial_transforms_probs_[0] <= class_prob < self.spatial_transforms_probs_[1]
        do_crop = self.spatial_transforms_probs_[1] <= class_prob < self.spatial_transforms_probs_[2]

        if do_expand:
            img_height, img_width = image.shape[:2]

            expand_ratio = np.random.uniform(1.0, self.max_expand_ratio_)

            expanded_height = int(img_height * expand_ratio)
            expanded_width = int(img_width * expand_ratio)
            h_off = int(np.floor(np.random.uniform(0., float(expanded_height - img_height))))
            w_off = int(np.floor(np.random.uniform(0., float(expanded_width - img_width))))

            expand_shift = (h_off, w_off)

            expand_fill = np.random.randint(0, 4) if self.expand_rand_fill_ else 0

        if do_crop:
            valid_objects = [obj for obj in objects_list if obj.class_id != self._ignore_class_id]

            if len(valid_objects) == 0:
                support_bbox = _rand_bbox()
            else:
                if len(valid_objects) == 1:
                    support_bbox = valid_objects[0]
                else:
                    objects_by_class = {}
                    for obj in valid_objects:
                        objects_by_class[obj.class_id] = objects_by_class.get(obj.class_id, []) + [obj]

                    center_class = np.random.choice(list(objects_by_class), 1, replace=False)[0]
                    num_center_objects = len(objects_by_class[center_class])
                    center_obj_id = np.random.randint(0, num_center_objects)
                    center_obj = objects_by_class[center_class][center_obj_id]

                    neighbours = [(obj, _bbox_dist(center_obj, obj)) for obj in valid_objects]
                    neighbours.sort(key=lambda t: t[1])

                    support_size = np.random.randint(2, len(valid_objects) + 1)
                    support_set = [neighbours[i][0] for i in range(support_size)]
                    support_bbox = _find_limits(support_set)

            crop_aspect_ratio = np.random.uniform(self.crop_ratio_limits_[0], self.crop_ratio_limits_[1])

        do_mirror = self.mirror_ and np.random.uniform(0.0, 1.0) < self.max_mirror_prob_

        glob_param = TransformParams(crop=do_crop,
                                     support_bbox=support_bbox if do_crop else None,
                                     aspect_ratio=crop_aspect_ratio if do_crop else None,
                                     mirror=do_mirror,
                                     expand=do_expand,
                                     expand_ratio=expand_ratio if do_expand else None,
                                     expand_shift=expand_shift if do_expand else None,
                                     expand_fill=expand_fill if do_expand else None)

        return glob_param

    def _transform_image_with_objects(self, img, objects, transform, trg_height, trg_width):
        """Carry out random transformation of input image with annotation according
           the transformation parameters.

        :param img: Input image
        :param objects: Annottaion
        :param transform: Parameters of transformations
        :param trg_height: Target image height
        :param trg_width: Target image width
        :return: Transformed image and its annotation
        """

        def _fit_bbox(src_bbox, trg_ratio, frame_size, delta_factor):
            """Fit input bounding box to the specified restrictions on aspect ratio and
               frame size.

            :param src_bbox: Input bounding box
            :param trg_ratio: Output aspect ratio of bounding box
            :param frame_size: Input frame sizes
            :param delta_factor: Scale to sample bounding box
            :return: Valid bounding box
            """

            out_h = src_bbox.ymax - src_bbox.ymin
            out_w = src_bbox.xmax - src_bbox.xmin
            src_aspect_ratio = float(out_h) / float(out_w)

            if src_aspect_ratio > trg_ratio:
                out_h = out_h
                out_w = out_h / trg_ratio
            else:
                out_h = out_w * trg_ratio
                out_w = out_w

            delta_x = delta_factor * out_w
            delta_y = delta_factor * out_h

            center_x = src_bbox.xmin + 0.5 * out_w + np.random.uniform(-delta_x, delta_x)
            center_y = src_bbox.ymin + 0.5 * out_h + np.random.uniform(-delta_y, delta_y)

            out_xmin = np.maximum(0, int((center_x - 0.5 * out_w) * frame_size[1]))
            out_ymin = np.maximum(0, int((center_y - 0.5 * out_h) * frame_size[0]))
            out_xmax = np.minimum(int((center_x + 0.5 * out_w) * frame_size[1]), frame_size[1])
            out_ymax = np.minimum(int((center_y + 0.5 * out_h) * frame_size[0]), frame_size[0])

            return [out_xmin, out_ymin, out_xmax, out_ymax]

        if transform is None:
            return cv2.resize(img, (trg_width, trg_height)), objects

        try:
            augmented_img = img
            augmented_objects = objects

            if transform.expand:
                expanded_height = int(augmented_img.shape[0] * transform. expand_ratio)
                expanded_width = int(augmented_img.shape[1] * transform.expand_ratio)

                if transform.expand_fill == 0:
                    expanded_img = np.zeros([expanded_height, expanded_width, 3], dtype=np.uint8)
                elif transform.expand_fill == 1:
                    color = np.array([np.random.randint(0, 256)] * 3, dtype=np.uint8)
                    expanded_img = np.full([expanded_height, expanded_width, 3], color, dtype=np.uint8)
                elif transform.expand_fill == 2:
                    color = np.random.randint(0, 256, 3, dtype=np.uint8)
                    expanded_img = np.full([expanded_height, expanded_width, 3], color, dtype=np.uint8)
                else:
                    expanded_img = np.random.randint(0, 256, [expanded_height, expanded_width, 3], dtype=np.uint8)

                roi_xmin = transform.expand_shift[1]
                roi_ymin = transform.expand_shift[0]
                roi_xmax = roi_xmin + augmented_img.shape[1]
                roi_ymax = roi_ymin + augmented_img.shape[0]

                expanded_img[roi_ymin:roi_ymax, roi_xmin:roi_xmax] = augmented_img
                augmented_img = expanded_img

                expand_scale = 1.0 / transform.expand_ratio
                expand_shift = (float(transform.expand_shift[0]) / float(expanded_height),
                                float(transform.expand_shift[1]) / float(expanded_width))

                expanded_objects = []
                for obj in augmented_objects:
                    expanded_objects.append(BBox(track_id=obj.track_id,
                                                 class_id=obj.class_id,
                                                 xmin=expand_shift[1] + expand_scale * obj.xmin,
                                                 ymin=expand_shift[0] + expand_scale * obj.ymin,
                                                 xmax=expand_shift[1] + expand_scale * obj.xmax,
                                                 ymax=expand_shift[0] + expand_scale * obj.ymax,
                                                 occluded=obj.occluded))
                augmented_objects = expanded_objects

            if transform.crop:
                src_height, src_width = augmented_img.shape[:2]
                crop_bbox = _fit_bbox(transform.support_bbox, transform.aspect_ratio, [src_height, src_width],
                                      delta_factor=self.crop_center_fraction_)

                crop_height = crop_bbox[3] - crop_bbox[1]
                crop_width = crop_bbox[2] - crop_bbox[0]

                augmented_img = augmented_img[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
                augmented_img = cv2.resize(augmented_img, (trg_width, trg_height))

                cropped_objects = []
                for obj in augmented_objects:
                    obj_xmin = np.maximum(0, int(obj.xmin * src_width)) - crop_bbox[0]
                    obj_ymin = np.maximum(0, int(obj.ymin * src_height)) - crop_bbox[1]
                    obj_xmax = np.minimum(int(obj.xmax * src_width), src_width) - crop_bbox[0]
                    obj_ymax = np.minimum(int(obj.ymax * src_height), src_height) - crop_bbox[1]

                    if obj_xmin < 0 and obj_xmax > crop_width and obj_ymin < 0 and obj_ymax > crop_height or \
                       obj_xmax <= 0 or obj_ymax <= 0 or obj_xmin >= crop_width or obj_ymin >= crop_height:
                        continue

                    out_obj_xmin = float(np.maximum(0, obj_xmin)) / float(crop_width)
                    out_obj_ymin = float(np.maximum(0, obj_ymin)) / float(crop_height)
                    out_obj_xmax = float(np.minimum(obj_xmax, crop_width)) / float(crop_width)
                    out_obj_ymax = float(np.minimum(obj_ymax, crop_height)) / float(crop_height)

                    out_obj_height = out_obj_ymax - out_obj_ymin
                    out_obj_width = out_obj_xmax - out_obj_xmin

                    if out_obj_height < self.min_bbox_size_ or out_obj_width < self.min_bbox_size_:
                        continue

                    cropped_objects.append(BBox(track_id=obj.track_id,
                                                class_id=obj.class_id,
                                                xmin=out_obj_xmin, ymin=out_obj_ymin,
                                                xmax=out_obj_xmax, ymax=out_obj_ymax,
                                                occluded=obj.occluded))
                augmented_objects = cropped_objects

            if augmented_img.shape[:2] != (trg_height, trg_width):
                augmented_img = cv2.resize(augmented_img, (trg_width, trg_height))

            if transform.mirror:
                augmented_img = augmented_img[:, ::-1, :]

                mirrored_objects = []
                for obj in augmented_objects:
                    mirrored_objects.append(BBox(track_id=obj.track_id,
                                                 class_id=obj.class_id,
                                                 xmin=1.0 - obj.xmax, ymin=obj.ymin,
                                                 xmax=1.0 - obj.xmin, ymax=obj.ymax,
                                                 occluded=obj.occluded))

                augmented_objects = mirrored_objects
        except Exception:
            return cv2.resize(img, (trg_width, trg_height)), objects

        return augmented_img, augmented_objects

    def _augment_image(self, img):
        """Carry out augmentation of image.

           Maintainable types of single image augmentation:
               * Blur
               * Gamma
               * Brightness
               * Down- and Up-Scale
               * Gaussian noise
               * Salt and Pepper

        :param img: Input image
        :return: Augmented image
        """

        augmented_img = img

        if self.blur_ and np.random.uniform(0.0, 1.0) < self.max_blur_prob_:
            filter_size = np.random.uniform(low=self.sigma_limits_[0], high=self.sigma_limits_[1])
            augmented_img[:, :, 0] = gaussian_filter(augmented_img[:, :, 0], sigma=filter_size)
            augmented_img[:, :, 1] = gaussian_filter(augmented_img[:, :, 1], sigma=filter_size)
            augmented_img[:, :, 2] = gaussian_filter(augmented_img[:, :, 2], sigma=filter_size)

        if self.gamma_ and np.random.uniform(0.0, 1.0) < self.max_gamma_prob_:
            rand_val = np.random.uniform(-self.delta_, self.delta_)
            gamma = np.log(0.5 + (2 ** (-0.5)) * rand_val) / np.log(0.5 - (2 ** (-0.5)) * rand_val)

            float_image = augmented_img.astype(np.float32) * (1. / 255.)
            augmented_img = (np.power(float_image, gamma) * 255.0).astype(np.int32)
            augmented_img[augmented_img > 255] = 255
            augmented_img[augmented_img < 0] = 0
            augmented_img = augmented_img.astype(np.uint8)

        if self.brightness_ and np.random.uniform(0.0, 1.0) < self.max_brightness_prob_:
            if np.average(augmented_img) > self.min_pos_:
                alpha = np.random.uniform(self.pos_alpha_[0], self.pos_alpha_[1])
                beta = np.random.randint(self.pos_beta_[0], self.pos_beta_[1])
            else:
                alpha = np.random.uniform(self.neg_alpha_[0], self.neg_alpha_[1])
                beta = np.random.randint(self.neg_beta_[0], self.neg_beta_[1])

            augmented_img = (augmented_img.astype(np.float32) * alpha + beta).astype(np.int32)
            augmented_img[augmented_img > 255] = 255
            augmented_img[augmented_img < 0] = 0
            augmented_img = augmented_img.astype(np.uint8)

        if self.down_up_scale_ and np.random.uniform(0.0, 1.0) < self.down_up_scale_prob_:
            src_height, src_width = augmented_img.shape[:2]

            scale_factor = np.random.uniform(self.min_scale_, 1.0)

            aug_height = int(src_height * scale_factor)
            aug_width = int(src_width * scale_factor)

            augmented_img = cv2.resize(augmented_img, (aug_width, aug_height))
            augmented_img = cv2.resize(augmented_img, (src_width, src_height))

        if self.noise_ and np.random.uniform(0.0, 1.0) < self.noise_prob_:
            noise_scale = np.random.uniform(0.0, self.noise_max_scale_) * 255.0

            augmented_img = augmented_img.astype(np.float32) + np.random.normal(0.0, noise_scale, augmented_img.shape)
            augmented_img[augmented_img < 0.0] = 0.0
            augmented_img[augmented_img > 255.0] = 255.0
            augmented_img = augmented_img.astype(np.uint8)

        if self.salt_pepper_ and np.random.uniform(0.0, 1.0) < self.salt_pepper_prob_:
            augmented_img[np.less(np.random.uniform(0.0, 1.0, augmented_img.shape), self.salt_pepper_p_)] = 0
            augmented_img[np.less(np.random.uniform(0.0, 1.0, augmented_img.shape), self.salt_pepper_p_)] = 255

        return augmented_img.astype(np.uint8)

    def _sample_annotated_image(self, frame_id):
        """Loads image from disk and augments it.

        :param frame_id: ID of loaded image
        :return: Image and its annotation
        """

        image, objects = self._data_sampler.get_frame_with_annotation(frame_id)
        assert image is not None
        assert objects is not None

        transform_params = self._sample_params(image, objects)
        transformed_image, transformed_objects = \
            self._transform_image_with_objects(image, objects, transform_params, self.height_, self.width_)

        augmented_image = self._augment_image(transformed_image)

        return transformed_objects, augmented_image

    def _sample_next_batch(self):
        """Generates next batch of images with annotation

        :return: Pair of images and its annotation
        """

        images_blob = []
        labels_blob = []
        batch_frame_ids = []
        item_id = 0
        while item_id < self.batch_size_:
            frame_id, objects, augmented_image = self.annotated_images_queue.get(True)
            if frame_id in batch_frame_ids:
                continue

            labels_blob += self._objects_to_blob(item_id, objects)
            images_blob.append(self._image_to_blob(augmented_image, self.width_, self.height_))
            batch_frame_ids.append(frame_id)

            item_id += 1

        images_blob = np.array(images_blob, dtype=np.float32)
        labels_blob = np.array(labels_blob, dtype=np.float32).reshape([1, 1, -1, 8])

        return images_blob, labels_blob

    def _set_data(self, data_sampler):
        """Sets loader of images.

        :param data_sampler: owner of loaded images
        """

        self._data_sampler = data_sampler

    def _load_params(self, param_str):
        """Loads layer parameters.

        :param param_str: Input str of parameters
        """

        layer_params = json.load(open(eval(param_str)['config']))

        assert 'tasks' in layer_params
        assert exists(layer_params['tasks'])
        assert 'batch' in layer_params
        assert 'height' in layer_params
        assert 'width' in layer_params
        assert 'valid_class_ids' in layer_params
        assert 'ignore_class_id' in layer_params

        self._valid_class_ids = layer_params['valid_class_ids']
        assert len(self._valid_class_ids) > 0

        self.use_attribute = layer_params['use_attribute']
        self.class_names_map = layer_params['class_names_map']
        assert self.class_names_map is not None

        self._ignore_class_id = layer_params['ignore_class_id']
        self._ignore_occluded = layer_params['ignore_occluded'] if 'ignore_occluded' in layer_params else True
        data_sampler = SampleDataFromDisk(layer_params['tasks'], self._ignore_occluded, self.class_names_map,
                                          self._valid_class_ids, self._ignore_class_id, self.use_attribute)
        self.batch_size_ = layer_params['batch']
        self._set_data(data_sampler)

        self.height_ = layer_params['height']
        self.width_ = layer_params['width']

        self.num_data_fillers_ = layer_params['num_data_fillers'] if 'num_data_fillers' in layer_params else 3
        self.data_queue_size_ = layer_params['data_queue_size'] if 'data_queue_size' in layer_params else 30
        self.single_iter_ = layer_params['single_iter'] if 'single_iter' in layer_params else False
        if self.single_iter_:
            assert self.num_data_fillers_ == 1

        self.blur_ = layer_params['blur'] if 'blur' in layer_params else False
        if self.blur_:
            self.sigma_limits_ = layer_params['sigma_limits'] if 'sigma_limits' in layer_params else [0.0, 0.5]
            self.max_blur_prob_ = layer_params['max_blur_prob'] if 'max_blur_prob' in layer_params else 0.5
            assert 0.0 <= self.sigma_limits_[0] < self.sigma_limits_[1]
            assert 0.0 <= self.max_blur_prob_ <= 1.0

        self.gamma_ = layer_params['gamma'] if 'gamma' in layer_params else False
        if self.gamma_:
            self.delta_ = layer_params['delta'] if 'delta' in layer_params else 0.15
            self.max_gamma_prob_ = layer_params['max_gamma_prob'] if 'max_gamma_prob' in layer_params else 0.5
            assert 0.0 < self.delta_ < 1.0
            assert 0.0 <= self.max_gamma_prob_ <= 1.0

        self.brightness_ = layer_params['brightness'] if 'brightness' in layer_params else False
        if self.brightness_:
            self.min_pos_ = layer_params['min_pos'] if 'min_pos' in layer_params else 100.0
            self.pos_alpha_ = layer_params['pos_alpha'] if 'pos_alpha' in layer_params else [0.2, 1.5]
            self.pos_beta_ = layer_params['pos_beta'] if 'pos_beta' in layer_params else [-100.0, 50.0]
            self.neg_alpha_ = layer_params['neg_alpha'] if 'neg_alpha' in layer_params else [0.9, 1.5]
            self.neg_beta_ = layer_params['neg_beta'] if 'neg_beta' in layer_params else [-20.0, 50.0]
            self.max_brightness_prob_ = layer_params[
                'max_brightness_prob'] if 'max_brightness_prob' in layer_params else 0.5
            assert 0.0 <= self.max_brightness_prob_ <= 1.0

        self.down_up_scale_ = layer_params['down_up_scale'] if 'down_up_scale' in layer_params else False
        if self.down_up_scale_:
            self.min_scale_ = layer_params['min_scale'] if 'min_scale' in layer_params else 0.4
            self.down_up_scale_prob_ =\
                layer_params['down_up_scale_prob'] if 'down_up_scale_prob' in layer_params else 0.1
            assert 0.0 < self.min_scale_ < 1.0
            assert 0.0 <= self.down_up_scale_prob_ <= 1.0

        self.noise_ = layer_params['noise'] if 'noise' in layer_params else False
        if self.noise_:
            self.noise_max_scale_ = layer_params['noise_max_scale'] if 'noise_max_scale' in layer_params else 0.1
            self.noise_prob_ = layer_params['noise_prob'] if 'noise_prob' in layer_params else 0.01
            assert self.noise_max_scale_ > 0.0
            assert 0.0 <= self.noise_prob_ <= 1.0

        self.salt_pepper_ = layer_params['salt_pepper'] if 'salt_pepper' in layer_params else False
        if self.salt_pepper_:
            self.salt_pepper_p_ = layer_params['salt_pepper_p'] if 'salt_pepper_p' in layer_params else 0.01
            self.salt_pepper_prob_ = layer_params['salt_pepper_prob'] if 'salt_pepper_prob' in layer_params else 0.01
            assert 0.0 < self.salt_pepper_p_ < 1.0
            assert 0.0 <= self.noise_prob_ <= 1.0

        self.mirror_ = layer_params['mirror'] if 'mirror' in layer_params else False
        if self.mirror_:
            self.max_mirror_prob_ = layer_params['max_mirror_prob'] if 'max_mirror_prob' in layer_params else 0.5
            assert 0.0 <= self.max_mirror_prob_ <= 1.0

        free_weight = float(layer_params['free_weight']) if 'free_weight' in layer_params else 0.1
        assert 0.0 <= free_weight <= 1.0
        expand_weight = float(layer_params['expand_weight']) if 'expand_weight' in layer_params else 1.0
        assert 0.0 <= expand_weight <= 1.0
        crop_weight = float(layer_params['crop_weight']) if 'crop_weight' in layer_params else 1.0
        assert 0.0 <= crop_weight <= 1.0
        spatial_transform_probs = np.cumsum([0.0, expand_weight, crop_weight, free_weight])
        self.spatial_transforms_probs_ = spatial_transform_probs / spatial_transform_probs[-1]

        self.max_expand_ratio_ = layer_params['max_expand_ratio'] if 'max_expand_ratio' in layer_params else 4.0
        assert self.max_expand_ratio_ > 1.0
        self.expand_rand_fill_ = layer_params['expand_rand_fill'] if 'expand_rand_fill' in layer_params else True

        self.crop_ratio_limits_ = layer_params['crop_ratio_limits']\
            if 'crop_ratio_limits' in layer_params else [0.5, 2.0]
        assert len(self.crop_ratio_limits_) == 2
        assert self.crop_ratio_limits_[0] < self.crop_ratio_limits_[1]
        self.crop_min_set_size_ = layer_params['crop_min_set_size'] \
            if 'crop_min_set_size' in layer_params else 4
        assert self.crop_min_set_size_ > 0
        self.crop_center_fraction_ = layer_params['crop_center_fraction']\
            if 'crop_center_fraction' in layer_params else 0.1
        assert 0.0 < self.crop_center_fraction_ < 1.0
        self.min_bbox_size_ = layer_params['min_bbox_size'] if 'min_bbox_size' in layer_params else 0.01
        assert 0.0 < self.min_bbox_size_ < 1.0

    @staticmethod
    def _ids_queue_filler(ids_queue, data_sampler, single_run):
        """Worker to fill queue of valid image IDs.

        :param ids_queue: Target queue to fill in
        :param data_sampler: Owner of loaded images
        """

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        while True:
            if ids_queue.empty():
                class_queues = data_sampler.get_class_frame_ids()
                min_queue_size = np.min([len(q) for q in itervalues(class_queues)])

                class_subsets = []
                for label in class_queues:
                    class_frame_ids = np.copy(class_queues[label])
                    subset_ids = np.random.choice(class_frame_ids, min_queue_size, replace=False)
                    class_subsets.append(subset_ids)

                final_num_classes = len(class_subsets)
                final_ids = np.zeros([final_num_classes * min_queue_size], dtype=np.int32)
                for i in range(final_num_classes):
                    final_ids[i::final_num_classes] = class_subsets[i]

                for i in final_ids:
                    ids_queue.put(i, True)
            else:
                if single_run:
                    break
                else:
                    time.sleep(10)

    def _data_queue_filler(self, data_queue, ids_queue, max_size, single_run):
        """Worker to fill queue of augmented images.

        :param data_queue: Target queue to fill in
        :param ids_queue: Queue if image IDs
        :param max_size: Maximal number of images in the queue
        """

        def _step():
            frame_id = ids_queue.get(True)

            objects, image = self._sample_annotated_image(frame_id)
            data_queue.put((frame_id, objects, image), True)

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if single_run:
            for _ in range(max_size):
                _step()
        else:
            while True:
                if data_queue.qsize() <= max_size:
                    _step()
                else:
                    time.sleep(1)

    def _start_prefetch(self):
        """Starts workers to fill queue of augmented images"""

        self.frame_ids_queue = Queue()
        self.annotated_images_queue = Queue()

        self._ids_filler_process = Process(target=self._ids_queue_filler,
                                           args=(self.frame_ids_queue, self._data_sampler, self.single_iter_))
        self._ids_filler_process.daemon = True
        self._ids_filler_process.start()

        self._data_fillers_pool = []
        for _ in range(self.num_data_fillers_):
            new_data_filler = Process(target=self._data_queue_filler,
                                      args=(self.annotated_images_queue, self.frame_ids_queue,
                                            self.data_queue_size_, self.single_iter_))
            new_data_filler.daemon = True
            new_data_filler.start()
            self._data_fillers_pool.append(new_data_filler)

    def setup(self, bottom, top):
        """Initializes layer.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        try:
            self._load_params(self.param_str)
            self._start_prefetch()
        except Exception:
            LOG('DataLayer exception: {}'.format(traceback.format_exc()))
            exit()

    def forward(self, bottom, top):
        """Carry out forward pass.

        :param bottom: List of bottom blobs
        :param top: List of top blobs
        """

        try:
            assert len(top) == 2

            images_blob, labels_blob = self._sample_next_batch()
            top[0].data[...] = images_blob

            top[1].reshape(1, 1, labels_blob.shape[2], 8)
            top[1].data[...] = labels_blob
        except Exception:
            LOG('DataLayer exception: {}'.format(traceback.format_exc()))
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

        top[0].reshape(self.batch_size_, 3, self.height_, self.width_)
        top[1].reshape(1, 1, 1, 8)

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


import argparse
import pickle
import os
import sys
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # pylint: disable=import-error

from utils.detection import evaluate_detections, voc_ap, miss_rate
from utils.dataset_xml_reader import read_annotation, convert_object_info, is_empty_bbox


# pylint: disable=unnecessary-lambda,invalid-name
def plot_curves(curves):
    """
    Plot curves
    """
    colors = sns.color_palette("hls", len(list(curves.keys())))
    plt.style.use("ggplot")
    plt.figure()
    i = 0
    for curve_name, curve_points in curves.items():
        if len(curve_points[0]) > 1:
            plt.plot(curve_points[1],
                     curve_points[0],
                     linewidth=2,
                     label=curve_name,
                     color=colors[i])
            i += 1
        else:
            print(curve_points)
            x, y = curve_points
            # Handle point of plotting view.
            x = max(x, 1.1e-4)
            y = max(y, 1e-5)
            plt.plot(x, y, linewidth=2, marker='D', markersize=10, label=curve_name)
    plt.xlim(1e-4, 1e4)
    plt.xscale("log", nonposx="clip", subsx=[])
    plt.xlabel(r"false positives per image", fontsize=12)
    y_range = np.concatenate((np.array([0.03, 0.05]),
                              np.arange(0.1, 0.55, 0.1),
                              np.array([0.64, 0.8, 1])))
    plt.yscale("log", nonposy="clip", subsy=[])
    plt.yticks(y_range, ["{:.2f}".format(i) for i in y_range])
    plt.ylabel(r"miss rate", fontsize=12)
    legend = plt.legend(loc="best", prop={'size': 8})
    legend.get_frame().set_alpha(0.5)


def plot_pr_curves(curves):
    """
    Plot precision/recall curves
    """
    colors = sns.color_palette("hls", len(list(curves.keys())))
    plt.style.use("ggplot")
    plt.figure()
    i = 0
    for curve_name, curve_points in curves.items():
        if len(curve_points[0]) > 1:
            plt.plot(curve_points[1],
                     curve_points[0],
                     linewidth=2,
                     label=curve_name,
                     color=colors[i])
            i += 1
        else:
            print(curve_points)
            plt.plot(curve_points[1],
                     curve_points[0],
                     linewidth=2,
                     marker='D',
                     markersize=10,
                     label=curve_name)
    plt.xlim(-0.05, 1.05)
    plt.xlabel(r"recall", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.ylabel(r"precision", fontsize=12)
    legend = plt.legend(loc="best", prop={'size': 8})
    legend.get_frame().set_alpha(0.5)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--det", dest="detections_file_path", nargs="+", required=True,
                        help="File with detector output.")
    parser.add_argument("--gt", dest="ground_truth_file_path", nargs="+", required=True,
                        help="File with ground truth bounding boxes.")
    parser.add_argument("--r", "--reasonable", dest="is_reasonable_subset", action="store_true",
                        help="Apply filter to ground truth: clip bounding boxes on image borders \
                             (according to `imsize` parameter values) and treat bounding boxes with height \
                             out of `objsize` range as ignored.")
    parser.add_argument("--v", "--visible", dest="is_visible_subset", action="store_true",
                        help="Apply filter to ground truth: delete bounding boxes with visible \
                             tags other than visible.")
    parser.add_argument("--f", "--filter", dest="do_detections_filtering", action="store_true",
                        help="Apply filter to detector output: remove bounding boxes with height "
                             "out of `objsize` range.")
    parser.add_argument("--im", "--imsize", dest="image_size", nargs=2,
                        type=int, default=(1920, 1080),
                        help="Image resolution. Used for filtering.")
    parser.add_argument("--obj", "--objsize", dest="object_size", nargs=2,
                        type=int, default=(10, 600),
                        help="Viable object height range. Used for filtering.")
    parser.add_argument("--o", "--result", dest="result_file_path", default="",
                        help="Path to file to save results to.")
    parser.add_argument("--show", dest="show_plots", action="store_true",
                        help="Show plots with quality curves.")
    parser.add_argument("--mm", "--multiple_matches",
                        dest="allow_multiple_matches", action="store_true",
                        help="Allow multiple matches per one ignored ground truth bounding box.")
    parser.add_argument("--c", "--class_lbl", dest="class_lbl", type=str, default="pedestrian",
                        help="Target class.")
    parser.add_argument("--s", "--square", dest="treat_as_square", action="store_true",
                        help="Treat object sizes as sqrt from square.")
    return parser.parse_args()


# pylint: disable=assignment-from-no-return,invalid-name
def clip_bbox(bbox, im_size=(1920, 1080)):
    """
    Clip bbox
    """
    bbox = np.maximum(np.copy(bbox), 0)
    x, y, w, h = bbox
    w = min(x + w, im_size[0]) - x
    h = min(y + h, im_size[1]) - y
    if w == 0 and h == 0:
        x = y = w = h = -1
    return np.array([x, y, w, h])


def reasonable_filter(annotation, height_range=(0, sys.maxsize),
                      image_size=(1920, 1080), class_lbl='pedestrian', treat_as_square=False):
    """
    Apply a reasonable filter
    """
    for image_annotation in list(annotation.values()):
        for object_annotation in image_annotation:
            bbox = object_annotation["bbox"]
            bbox = clip_bbox(bbox, image_size)

            check_value = bbox[3]
            if treat_as_square:
                check_value = math.sqrt(bbox[2]*bbox[3])

            object_annotation["is_ignored"] = object_annotation.get("is_ignored", False) or \
                                              not object_annotation.get("visibility", True) or \
                                              not (height_range[0] <= check_value <= height_range[1]) or \
                                              is_empty_bbox(*bbox) or \
                                              object_annotation.get("type", "") != class_lbl

            object_annotation["bbox"] = bbox

        for ignore_reg in image_annotation.ignore_regs:
            image_annotation.objects.append({"is_ignored": True,
                                             "bbox": [ignore_reg[0], ignore_reg[1],
                                                      ignore_reg[0] + ignore_reg[2],
                                                      ignore_reg[1] + ignore_reg[3]]})

    return annotation


def filter_annotation(annotation, height_range=(0, sys.maxsize),
                      image_size=(1920, 1080), treat_as_square=False):
    """
    Filter annotation
    """
    for key, image_annotation in annotation.items():
        for object_annotation in image_annotation:
            bbox = clip_bbox(object_annotation["bbox"], image_size)

            check_value = bbox[3]
            if treat_as_square:
                check_value = math.sqrt(bbox[2]*bbox[3])

            if not (height_range[0] <= check_value <= height_range[1]) or is_empty_bbox(*bbox):
                bbox = None
            object_annotation["bbox"] = bbox

        annotation[key].objects = [object_annotation for object_annotation in image_annotation
                                   if object_annotation["bbox"] is not None]
    return annotation


def count_objects(annotation):
    """
    Get number of objects in annotations
    """
    n = 0
    for image_annotation in list(annotation.values()):
        n += np.count_nonzero([not bool(object_annotation.get("is_ignored", False))
                               for object_annotation in image_annotation])
    return n


def size_stats(annotation, class_lbl):
    """
    Plot statistics for class
    :param class_lbl: class label
    """
    widths = []
    heights = []
    for image_annotation in list(annotation.values()):
        for object_annotation in image_annotation:
            if not object_annotation.get("is_ignored", False):
                bbox = object_annotation["bbox"]
                if object_annotation["type"] == class_lbl:
                    widths.append(bbox[2])
                    heights.append(bbox[3])

    widths = np.array(widths)
    heights = np.array(heights)

    print("Stats:")
    print("widths: ", np.mean(widths), np.std(widths), np.min(widths), np.max(widths))
    print("heights: ", np.mean(heights), np.std(heights), np.min(heights), np.max(heights))

    plt.hist(heights, bins=500)
    plt.show()


# pylint: disable=len-as-condition
def main():
    """
    Evaluate found detections
    """
    args = parse_args()

    transformers = {"id": int, "bbox": lambda x: list(map(float, x.strip().split())), "score": float,
                    "is_ignored": lambda x: bool(int(x)), "visibility": lambda x: not (
                        x == "partially occluded" or x == "heavy occluded") if args.is_visible_subset else True}

    ground_truth = {}
    for ground_truth_file_path in args.ground_truth_file_path:
        ground_truth.update(read_annotation(ground_truth_file_path))

    images_with_gt = set([image_annotation.image_path for image_annotation in ground_truth.values()])

    for image_annotation in list(ground_truth.values()):
        for object_annotation in image_annotation:
            object_annotation = convert_object_info(transformers, object_annotation)

    if args.is_reasonable_subset:
        ground_truth = reasonable_filter(ground_truth,
                                         args.object_size,
                                         args.image_size,
                                         args.class_lbl,
                                         args.treat_as_square)

    print("#frames = {}".format(len(ground_truth)))
    print("#gt objects = {}".format(count_objects(ground_truth)))

    miss_rate_plots = {}
    precision_plots = {}
    for detections_file_path in args.detections_file_path:
        t = time.time()
        detections = read_annotation(detections_file_path)
        print("Reading {} ms".format((time.time() - t) * 1e3))

        detections = {frame_id: image_annotation
                      for frame_id, image_annotation in detections.items()
                      if image_annotation.image_path in images_with_gt}
        assert len(detections) > 0, "No detections found for provided ground truth."

        t = time.time()
        for image_annotation in list(detections.values()):
            for object_annotation in image_annotation:
                object_annotation = convert_object_info(transformers, object_annotation)
        print("Converting {} ms".format((time.time() - t) * 1e3))

        if args.do_detections_filtering:
            detections = filter_annotation(detections, args.object_size, args.image_size, args.treat_as_square)

        t = time.time()

        recall, precision, miss_rates, fppis = evaluate_detections(list(ground_truth.values()),
                                                                   list(detections.values()),
                                                                   args.class_lbl,
                                                                   allow_multiple_matches_per_ignored=args.allow_multiple_matches,
                                                                   overlap_threshold=0.5)
        print("Calculating {} ms".format((time.time() - t) * 1e3))

        mr = miss_rate(miss_rates, fppis)
        ap = voc_ap(recall, precision)

        print("#frames = {}".format(len(ground_truth)))
        print("#gt objects = {}".format(count_objects(ground_truth)))
        print("#detections = {}".format(count_objects(detections)))
        print("miss rate at fppi0.1 = {:.2%}".format(mr))
        print("AP = {:.2%}".format(ap))

        if args.result_file_path:
            results = {"miss_rates": miss_rates,
                       "fppis": fppis,
                       "precisions": precision,
                       "recalls": recall,
                       "ap": ap,
                       "mr": mr}
            with open(args.result_file_path, "w") as f:
                pickle.dump(results, f)
            print("Results are dumped to '{}'.".format(args.result_file_path))

        miss_rate_plots[os.path.basename(detections_file_path)] = [miss_rates, fppis]
        precision_plots[os.path.basename(detections_file_path)] = [precision, recall]

    if args.show_plots:
        plot_curves(miss_rate_plots)
        plot_pr_curves(precision_plots)
        plt.show()


if __name__ == "__main__":
    main()

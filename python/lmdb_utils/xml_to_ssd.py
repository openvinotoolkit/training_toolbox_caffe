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

import os
import argparse
from tqdm import tqdm
import cv2
from lxml import etree

from utils.dataset_xml_reader import read_annotation

def make_dir(path):
    """ Create a directory """
    try:
        os.makedirs(path)
    except Exception:
        pass

# pylint: disable=redefined-argument-from-local
def indent(elem, level=0):
    """
    Add indentation
    """
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def process_frame_annotation(frame_annotation, image, root, size_thresh, is_train, scale_x, scale_y, show):
    """
    Parse image annotation
    """
    for frame_obj in frame_annotation:
        write_header = False
        for i in frame_obj.keys():
            if i == "bbox":
                bbox = frame_obj[i].split()
                xmin = float(bbox[0])
                ymin = float(bbox[1])
                width = float(bbox[2])
                height = float(bbox[3])
                xmax = xmin + width
                ymax = ymin + height
                xmin = xmin / scale_x
                ymin = ymin / scale_y
                xmax = xmax / scale_x
                ymax = ymax / scale_y

                if(is_train and (xmax - xmin < size_thresh or ymax - ymin < size_thresh)):
                    continue

                xmin = int(round(xmin - 0.49))
                ymin = int(round(ymin - 0.49))
                xmax = int(round(xmax + 0.49))
                ymax = int(round(ymax + 0.49))
                if xmax > image.shape[1] - 1:
                    xmax = image.shape[1] - 1
                if ymax > image.shape[0] - 1:
                    ymax = image.shape[0] - 1

                if not write_header:
                    obj = etree.SubElement(root, "object")
                    etree.SubElement(obj, "name").text = 'face'
                    etree.SubElement(obj, "difficult").text = "0"
                    write_header = True

                bndbox = etree.SubElement(obj, "bndbox")
                etree.SubElement(bndbox, "xmin").text = str(xmin)
                etree.SubElement(bndbox, "ymin").text = str(ymin)
                etree.SubElement(bndbox, "xmax").text = str(xmax)
                etree.SubElement(bndbox, "ymax").text = str(ymax)

                if show:
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (128, 0, 255))

def main():
    """
    Convert xml annotation to list of annotations per image appropriate for ssd
    """
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ssd_path', default='', type=str)
    parser.add_argument('--xml_path_train', default='', type=str)
    parser.add_argument('--xml_path_val', default='', type=str)
    parser.add_argument("--net_res", nargs=2, type=int, default=(300, 300), help="SSD input resolution (w h)")
    parser.add_argument("--size_thresh", type=int, default=8, help="Min object size in train")
    parser.add_argument("--show", type=bool, default=False, help="Show annotation")
    args = parser.parse_args()

    subsets = ['wider_train', 'wider_val']
    annotations = {subsets[0] : read_annotation(args.xml_path_train),
                   subsets[1] : read_annotation(args.xml_path_val)}

    image_width = args.net_res[0]
    image_height = args.net_res[1]

    for subset in subsets:
        annotations_folder = "annotations_{}/".format(subset)
        images_folder = "images_{}/".format(subset)
        list_out_name_ssd = os.path.join(args.ssd_path, "{}.txt".format(subset))
        list_out_name = os.path.join(args.ssd_path, "list_{}.txt".format(subset))
        list_out_name_size = os.path.join(args.ssd_path, "list_{}_size.txt".format(subset))

        make_dir(os.path.join(args.ssd_path, annotations_folder))
        make_dir(os.path.join(args.ssd_path, images_folder))
        file_out_ssd = open(list_out_name_ssd, "w")
        file_out = open(list_out_name, "w")
        file_out_size = open(list_out_name_size, "w")

        annotation = annotations[subset]

        for i in tqdm(range(len(annotation)), desc='Processing ' + subset):
            try:
                frame_annotation = annotation[i]
            except KeyError as ex:
                print ex

            if len(frame_annotation) == 0:
                continue

            root = etree.Element("annotation")
            size = etree.SubElement(root, "size")

            image_path = frame_annotation.image_path
            image_name = os.path.split(image_path)[1][:-3] + 'png'
            image_name_out = images_folder + image_name[:-3] + 'png'
            image_path_out = os.path.abspath(os.path.join(args.ssd_path, image_name_out))
            annotation_name = annotations_folder + image_name + ".xml"
            file_out_ssd.write("{} {}\n".format(image_name_out, annotation_name))
            file_out.write("{}\n".format(image_path_out))
            file_out_size.write("{} {} {}\n".format(image_name[:-4], image_height, image_width))

            image = cv2.imread(image_path)
            scale_x = float(image.shape[1]) / image_width
            scale_y = float(image.shape[0]) / image_height
            image = cv2.resize(image, (image_width, image_height))
            cv2.imwrite(image_path_out, image)

            etree.SubElement(size, "width").text = str(image_width)
            etree.SubElement(size, "height").text = str(image_height)

            process_frame_annotation(frame_annotation,
                                     image,
                                     root,
                                     args.size_thresh,
                                     subset == 'wider_train',
                                     scale_x,
                                     scale_y,
                                     args.show)
            eltree = etree.ElementTree(root)
            indent(eltree.getroot())
            eltree.write(os.path.join(args.ssd_path, annotation_name))

            if args.show:
                cv2.imshow("test", image)
                cv2.waitKey(1000)

if __name__ == '__main__':
    main()

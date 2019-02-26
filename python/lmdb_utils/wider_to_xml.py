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

import xml.etree.cElementTree as ET
import cv2
import math
import os
import numpy as np
import argparse

def make_dir(path):
  try:
    import os
    os.makedirs(path)
  except:
    pass

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def write_xml(image, image_name, bboxes):
        global path_out, show, root, object_num, image_num
        image_path_out = path_out + image_name

        image_elem = ET.SubElement(root, "image{:0=6}".format(image_num))
        ET.SubElement(image_elem, "path").text = image_path_out
        image_num += 1
        object_num = 0

        cv2.imwrite(image_path_out, image)

        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            object = ET.SubElement(image_elem, "object{:0=6}".format(object_num))
            ET.SubElement(object, "type").text = "face"
            ET.SubElement(object, "id").text = str(object_num)
            ET.SubElement(object, "bbox").text = "{} {} {} {}".format(xmin, ymin, xmax-xmin, ymax-ymin)
            object_num += 1

            if (show):
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (128, 0, 255))


def parse_args():
  parser = argparse.ArgumentParser(description='Convert wider annotations to xml format')
  parser.add_argument('root_path', help='Path to a root folder')
  parser.add_argument('images_path', help='Path to images')
  parser.add_argument('gt_path', help='Path to annotations')
  parser.add_argument('mode', help='Train or val')
  parser.add_argument("--show", type=bool, default=False, help='Show annotations')
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  path_root = args.root_path
  images_path = args.images_path
  gt_name = args.gt_path
  mode = args.mode
  show = args.show

  if mode == "train":
    path_out = path_root + "processed_train/"
    result_file_name = path_root + "wider_train.xml"
  else:
    path_out = path_root + "processed_val/"
    result_file_name = path_root + "wider_val.xml"
  #os.system("rm -rf " + path_out)
  make_dir(path_out)

  gt_file = open(gt_name, "r")
  lines = [line.rstrip('\n') for line in gt_file]

  root = ET.Element("opencv_storage")

  image_num = 0
  object_num = 0
  idx = 0
  while idx < len(lines):
    image_name = lines[idx]
    image_path = os.path.join(images_path, image_name)
    bbox_num = int(lines[idx+1])
    idx += 2

    image = cv2.imread(image_path)

    image_name_out = image_name.split('/')[-1]
    image_name_out = image_name_out[:-4] + ".png"

    image_path = images_path + image_name
    print(image_path, image_name, image_name_out)

    if (show):
        image = cv2.imread(image_path)
    print image.shape

    bboxes = []
    for obj_idx in xrange(bbox_num):
        attr = lines[idx].split()
        idx += 1

        x1 = float(attr[0])
        y1 = float(attr[1])
        w = float(attr[2])
        h = float(attr[3])
        blur = attr[4]
        expression = attr[5]
        illumination = attr[6]
        invalid = attr[7]
        occlusion = attr[8]
        pose = attr[9]

        xmin = x1
        ymin = y1
        xmax = (x1 + w)
        ymax = (y1 + h)

        bboxes.append((xmin, ymin, xmax, ymax))

    write_xml(image, image_name_out, bboxes)

    if (show):
        cv2.imshow("test", image)
        cv2.waitKey(2000)

  xml_doc = ET.ElementTree(root)
  indent(xml_doc.getroot())
  xml_doc.write(result_file_name)


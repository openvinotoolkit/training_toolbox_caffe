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

import re
import itertools
import six
import numpy as np
from tqdm import tqdm
from lxml import etree

# pylint: disable=old-style-class
class ImageAnnotation:
    """
    Class for image annotation
    """
    def __init__(self, image_path, objects=None, ignore_regs=None):
        self.image_path = image_path
        self.objects = objects if objects else []
        self.ignore_regs = ignore_regs if ignore_regs else []

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, item):
        return self.objects[item]

def bbox_to_string(bbox):
    """ Store bbox coordinated to string"""
    return ' '.join([str(int(float(coord))) for coord in bbox])

# pylint: disable=invalid-name
def is_empty_bbox(x, y, w, h):
    """ Check if the bbox is empty """
    bbox = np.asarray([x, y, w, h])
    return np.any(bbox == -1)


def write_annotation(annotation, filename, is_canonical=False):
    """
    Write annotation
    """
    root = etree.Element('opencv_storage')

    for frame in tqdm(sorted(annotation, key=lambda x: annotation[x].image_path), desc='Converting to ' + filename):
        image = etree.SubElement(root, 'image' + str(frame).zfill(6))

        if not is_canonical:
            image_path = etree.SubElement(image, 'path')
            image_path.text = annotation[frame].image_path

            if annotation[frame].ignore_regs:
                ignore_regions = etree.SubElement(image, 'ignore_reg')
                ignore_regions.text = " ".join(
                    map(str, [" ".join(map(str, val)) for val in annotation[frame].ignore_regs]))

        canonical_tags = ("bbox", "type", "quality", "id", "visibility")
        for obj_idx, obj in enumerate(annotation[frame]):
            obj_element = etree.SubElement(image, 'object' + str(obj.get("id", obj_idx)).zfill(6))

            for key, value in six.iteritems(obj):
                if not is_canonical or key in canonical_tags:
                    obj_feature = etree.SubElement(obj_element, key)
                    obj_feature.text = str(value)

    with open(filename, 'wb') as output:
        output.write(etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='utf-8'))


def read_object_info(xml_root):
    """
    Read objects
    """
    obj = {}
    tags = np.unique([element.tag for element in xml_root])
    for tag in tags:
        values = [x.text for x in xml_root.findall(tag)]
        if len(values) == 1:
            values = values[0]
        obj[tag] = values
    return obj


def convert_object_info(converters, obj_info):
    """
    Convert object information
    """
    for key, transform in converters.iteritems():
        if key in obj_info:
            obj_info[key] = transform(obj_info[key])
    return obj_info


def chunkwise(t, size=2):
    """ Get a chunk """
    it = iter(t)
    return itertools.izip(*[it]*size)


def read_regions(text):
    """ Read regions """
    if text is None:
        return None
    return [list(val) for val in list(chunkwise(map(int, text.split(" ")), 4))]


def read_annotation(filename):
    """
    Read annotation
    """
    tree = etree.parse(filename)
    root = tree.getroot()

    objects = {}
    for frame in tqdm(root, desc='Reading ' + filename):
        current_objects = []
        for obj in frame:
            if obj.tag.startswith("object"):
                current_objects.append(read_object_info(obj))

        path_tag = frame.find("path")
        if path_tag is not None:
            image_path = path_tag.text
        else:
            image_path = frame.tag

        ignore_reg_tag = frame.find("ignore_reg")
        if ignore_reg_tag is not None:
            ignore_regions = read_regions(ignore_reg_tag.text)
        else:
            ignore_regions = None

        frame_id = int(re.search(r"image([0-9]+)", frame.tag).group(1))
        objects[frame_id] = ImageAnnotation(image_path, current_objects, ignore_regions)

    return objects
